"""GMM-based Expert-Parallel MoE layer and weight mapping utilities."""

import math
import os
from functools import partial

import jax
from flax import nnx
from jax import numpy as jnp
from jax import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata
from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm

# Re-export for backward compatibility: external code imports from this module.
from sgl_jax.srt.layers.fused_moe import FusedEPMoE, FusedEPMoEV2  # noqa: F401
from sgl_jax.srt.layers.gate import GateLogit, TopK  # noqa: F401
from sgl_jax.srt.utils.profiling_utils import named_scope
from sgl_jax.srt.utils.quantization.quantization_utils import (
    quantize_tensor,
    quantize_tensor_simple,
)
from sgl_jax.srt.model_loader.weights import WeightSpec

# Opt-in: replace the 3-op unpermute (gather -> reshape/fp32 -> weighted sum)
# with the single fused SparseCore ragged_gather_reduce_v2 kernel ported from
# tpu-inference. Off by default so the stock path stays bit-identical.
_MOE_FUSED_UNPERMUTE_VERSION = "ragged-gather-reduce-v2-rec2"
_USE_FUSED_UNPERMUTE = os.environ.get("SGL_MOE_FUSED_UNPERMUTE", "0") == "1"
_USE_RAGGED_GATHER = os.environ.get("SGL_MOE_RAGGED_GATHER", "1") == "1"
_USE_EXPERT_SCATTER = os.environ.get("SGL_MOE_EXPERT_SCATTER", "1") == "1"
_MOE_FUSED_MIN_TOKENS = int(os.environ.get("SGL_MOE_FUSED_MIN_TOKENS", "4096") or "4096")
_MOE_FUSED_DEBUG = os.environ.get("SGL_MOE_FUSED_DEBUG", "0") == "1"
# Recommendation 2: number of chunks for pipelined ragged_gather_reduce_v2 + psum_scatter.
_MOE_CHUNK_STAGE = int(os.environ.get("SGL_MOE_CHUNK_STAGE", "4") or "4")


def _ranged_swiglu(
    w0: jax.Array,
    w1: jax.Array,
    token_start: jax.Array,
    token_end: jax.Array,
    block_m: int = 1024,
) -> jax.Array:
    """Compute silu(w0) * w1 in-place on rows [token_start, token_end) aligned to block_m."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    m, n = w0.shape
    num_blocks = m // block_m
    bounds = jnp.stack(
        [token_start.reshape(()), token_end.reshape(())]
    ).astype(jnp.int32)

    def _kernel(
        bounds_ref,
        w0_hbm_ref,
        w1_hbm_ref,
        out_hbm_ref,
        w0_vmem_ref,
        w1_vmem_ref,
        out_vmem_ref,
        sem_ref,
    ):
        t_start = bounds_ref[0]
        t_end = bounds_ref[1]
        b_start = jnp.clip(t_start // block_m, 0, num_blocks)
        b_end = jnp.clip(pl.cdiv(t_end, block_m), 0, num_blocks)

        def body(b_idx, _):
            row_offset = b_idx * block_m
            in_sem = sem_ref.at[0]
            out_sem = sem_ref.at[1]
            cp0 = pltpu.make_async_copy(
                w0_hbm_ref.at[pl.ds(row_offset, block_m)],
                w0_vmem_ref,
                in_sem,
            )
            cp1 = pltpu.make_async_copy(
                w1_hbm_ref.at[pl.ds(row_offset, block_m)],
                w1_vmem_ref,
                in_sem,
            )
            cp0.start()
            cp1.start()
            cp0.wait()
            cp1.wait()

            v0_f32 = w0_vmem_ref[...].astype(jnp.float32)
            act = (v0_f32 * jax.nn.sigmoid(v0_f32)).astype(w0_vmem_ref.dtype)
            out_vmem_ref[...] = jnp.multiply(act, w1_vmem_ref[...])

            cpo = pltpu.make_async_copy(
                out_vmem_ref,
                out_hbm_ref.at[pl.ds(row_offset, block_m)],
                out_sem,
            )
            cpo.start()
            cpo.wait()

        jax.lax.fori_loop(b_start, b_end, body, None)

    return pl.pallas_call(
        _kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            grid=(1,),
            scratch_shapes=[
                pltpu.VMEM((block_m, n), w0.dtype),
                pltpu.VMEM((block_m, n), w1.dtype),
                pltpu.VMEM((block_m, n), w0.dtype),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
        ),
        out_shape=jax.ShapeDtypeStruct(w0.shape, w0.dtype),
        input_output_aliases={1: 0},
        name="moe_ranged_swiglu",
    )(bounds, w0, w1)


class EPMoE(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        ep_size: int,
        mesh: Mesh,
        intermediate_dim: int = 2048,
        weight_dtype: jnp.dtype = jnp.bfloat16,
        dtype: jnp.dtype = jnp.bfloat16,
        activation: str = "silu",
        layer_id: int = 0,
        quantization_config=None,
        physical_to_logical_map: "jax.Array | None" = None,
        pre_gather_quant_dtype=None,
        moe_dp_size: int = 1,
        use_sc_permute: bool | None = None,
    ):
        self.num_experts_per_tok = num_experts_per_tok
        self.use_sc_permute = bool(use_sc_permute) if use_sc_permute is not None else False
        self.physical_to_logical_map = physical_to_logical_map
        self.pre_gather_quant_dtype = pre_gather_quant_dtype
        self.moe_dp_size = moe_dp_size
        self.replicate_experts = self.moe_dp_size > 1

        metadata = get_global_expert_location_metadata()
        if metadata is not None and layer_id is not None:
            self.num_experts = metadata.num_physical_experts
        else:
            self.num_experts = num_experts

        self.intermediate_dim = intermediate_dim
        self.weight_dtype = weight_dtype
        self.dtype = dtype  # original dtype
        self.layer_id = layer_id
        self.ep_size = ep_size
        self.original_mesh = mesh
        self.mesh = mesh
        self.activation = activation
        self.hidden_size = hidden_size

        # Get quantization settings from config
        self.quantized_dtype = (
            quantization_config.get_moe_weight_dtype() if quantization_config else None
        )
        self.activation_quantized_dtype = (
            quantization_config.get_moe_activation_dtype() if quantization_config else None
        )
        self.weight_block_size = (
            getattr(quantization_config, "weight_block_size", None) if quantization_config else None
        )

        if self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts({self.num_experts}) must be divisible by ep_size ({self.ep_size})"
            )
        world_size = math.prod(self.mesh.shape.values())
        self.tp_size = world_size // self.ep_size
        self.experts_per_device = self.num_experts // self.ep_size

        devices = self.mesh.devices.flatten()
        self.moe_mesh = jax.sharding.Mesh(
            devices.reshape(self.ep_size, self.tp_size),
            axis_names=("expert", "tensor"),
            axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
        )

        abstract_mesh = self.mesh.abstract_mesh
        self.updated_mesh = abstract_mesh.update(
            axis_sizes=(self.ep_size, self.tp_size), axis_names=("expert", "tensor")
        )

        with jax.sharding.use_abstract_mesh(self.updated_mesh):
            # MOE weights' shape is (num_experts, k, n)
            self.wi_0 = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (self.num_experts, hidden_size, intermediate_dim),
                    dtype=weight_dtype,
                    out_sharding=P("expert", None, "tensor"),
                )
            )

            self.wi_1 = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (self.num_experts, hidden_size, intermediate_dim),
                    dtype=weight_dtype,
                    out_sharding=P("expert", None, "tensor"),
                )
            )

            self.wo = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (self.num_experts, intermediate_dim, hidden_size),
                    dtype=weight_dtype,
                    out_sharding=P("expert", "tensor", None),
                )
            )

            # Scales are None by default - only set by quantize_weights() if quantization is enabled
            # gmm kernel handles None scales properly (no scaling applied)
            self.wi_0_scale = None
            self.wi_1_scale = None
            self.wo_scale = None

    def _detect_device_capabilities(self):
        try:
            devices = jax.devices()
            is_cpu_only = all(device.platform == "cpu" for device in devices)
            can_use_ragged = not is_cpu_only and hasattr(jax.lax, "ragged_all_to_all")

            device_types = [device.platform for device in devices]
            primary_device = device_types[0] if device_types else "unknown"

            return can_use_ragged, primary_device
        except Exception as _:
            return False, "cpu"

    def _normalize_scale_for_gmm(
        self,
        scale: jax.Array | None,
        weight: jax.Array,
        *,
        scale_name: str,
    ) -> jax.Array | None:
        """Normalize offline/runtime scale tensors to GMM's 4D layout.

        Accepted inputs intentionally cover the layouts we see in practice:

        - per-channel: ``[E, out_dim]``
        - already-kernel-ready: ``[E, k_blocks, 1, out_dim]``
        - sub-channel / block-channel: ``[E, out_dim, k_blocks]`` or
          ``[E, k_blocks, out_dim]``
        - offline 2D block quant: ``[E, out_blocks, k_blocks]``

        The returned tensor always matches the GMM contract
        ``[E, k_blocks, 1, out_dim]``.
        """
        if scale is None:
            return None

        # Weight layout is [E, k, n] where k=contraction dim, n=output dim.
        num_experts, in_dim, out_dim = weight.shape

        if scale.ndim == 4:
            if scale.shape[0] != num_experts or scale.shape[2] != 1 or scale.shape[3] != out_dim:
                raise ValueError(
                    f"Unsupported {scale_name} shape {scale.shape} for weight shape {weight.shape}. "
                    "Expected 4D GMM scale layout [E, k_blocks, 1, out_dim]."
                )
            if self.weight_block_size is None:
                if scale.shape[1] != 1:
                    raise ValueError(
                        f"Unsupported {scale_name} shape {scale.shape} for weight shape {weight.shape}. "
                        "Per-channel 4D GMM scales must have k_blocks=1."
                    )
            else:
                block_size_k = int(self.weight_block_size[1])
                expected_k_blocks = (in_dim + block_size_k - 1) // block_size_k
                if scale.shape[1] not in (1, expected_k_blocks):
                    raise ValueError(
                        f"Unsupported {scale_name} shape {scale.shape} for weight shape {weight.shape}. "
                        f"Expected k_blocks dimension to be 1 or {expected_k_blocks}."
                    )
            final_scale_sharding = (
                P("expert", None, None, None)
                if scale_name == "wo_scale"
                else P("expert", None, None, "tensor")
            )
            return jax.sharding.reshard(scale, final_scale_sharding)

        if scale.ndim == 2 and scale.shape == (num_experts, out_dim):
            return scale[:, None, None, :]

        if scale.ndim == 3:
            if scale.shape == (num_experts, 1, out_dim):
                return scale[:, :, None, :]

            # Support offline 2D block quant checkpoints whose scales are stored as
            # [num_experts, out_blocks, in_blocks]. GMM expects [E, k_blocks, 1, out_dim].
            if (
                self.weight_block_size is not None
                and isinstance(self.weight_block_size, (list, tuple))
                and len(self.weight_block_size) == 2
            ):
                block_size_out = int(self.weight_block_size[0])
                block_size_k = int(self.weight_block_size[1])
                expected_out_blocks = (out_dim + block_size_out - 1) // block_size_out
                expected_k_blocks = (in_dim + block_size_k - 1) // block_size_k

                if scale.shape == (num_experts, out_dim, expected_k_blocks):
                    final_scale_sharding = (
                        P("expert", None, None, None)
                        if scale_name == "wo_scale"
                        else P("expert", None, None, "tensor")
                    )
                    scale_gmm = jnp.transpose(scale, (0, 2, 1))[:, :, None, :]
                    return jax.sharding.reshard(scale_gmm, final_scale_sharding)

                if scale.shape == (num_experts, expected_out_blocks, expected_k_blocks):
                    scale_per_out_sharding = (
                        P("expert", None, None)
                        if scale_name == "wo_scale"
                        else P("expert", "tensor", None)
                    )
                    final_scale_sharding = (
                        P("expert", None, None, None)
                        if scale_name == "wo_scale"
                        else P("expert", None, None, "tensor")
                    )
                    out_block_ids = jnp.arange(out_dim, dtype=jnp.int32) // block_size_out
                    scale_per_out = scale.at[:, out_block_ids, :].get(
                        out_sharding=scale_per_out_sharding
                    )
                    scale_gmm = jnp.transpose(scale_per_out, (0, 2, 1))[:, :, None, :]
                    return jax.sharding.reshard(scale_gmm, final_scale_sharding)

                if scale.shape == (num_experts, expected_k_blocks, out_dim):
                    return scale[:, :, None, :]

        raise ValueError(
            f"Unsupported {scale_name} shape {scale.shape} for weight shape {weight.shape}. "
            "Expected one of: [E, out_dim], [E, 1, out_dim], [E, k_blocks, 1, out_dim], "
            "or offline block format [E, out_blocks, k_blocks]."
        )

    def quantize_weights(self, is_static: bool = False, *, abstract: bool = False):
        """Quantize MoE weights in-place or initialize params for static loading."""
        if self.quantized_dtype is None:
            return

        def _get_block_size_k(
            *,
            hidden_size: int,
            intermediate_dim: int,
            weight_block_size: list[int] | tuple[int, int] | None,
        ) -> int | None:
            """Extract the contracting-dimension block size for MoE weights.

            EPMoE only block-quantizes along the GEMM ``K`` dimension, so for a
            configured ``(block_n, block_k)`` we consume only ``block_k`` here.
            The divisibility checks keep the later GMM scale layout well-defined.
            """
            if weight_block_size is None:
                return None
            if not (isinstance(weight_block_size, (list, tuple)) and len(weight_block_size) == 2):
                raise ValueError(
                    f"EPMoE weight_block_size must be a 2-element list [block_n, block_k], "
                    f"got {weight_block_size}"
                )

            block_size_k = int(weight_block_size[1])
            if block_size_k <= 0:
                raise ValueError(f"EPMoE weight_block_size[1] must be > 0, got {block_size_k}")
            if hidden_size % block_size_k != 0:
                raise ValueError(
                    f"EPMoE hidden_size={hidden_size} not divisible by block_size_k={block_size_k}"
                )
            if intermediate_dim % block_size_k != 0:
                raise ValueError(
                    f"EPMoE intermediate_dim={intermediate_dim} not divisible by block_size_k={block_size_k}"
                )
            return block_size_k

        mesh_context = (
            jax.sharding.use_abstract_mesh(self.moe_mesh.abstract_mesh)
            if abstract
            else jax.set_mesh(self.moe_mesh)
        )
        with mesh_context:
            if is_static:
                for name in ("wi_0", "wi_1", "wo"):
                    param = getattr(self, name)
                    if isinstance(param.value, jax.ShapeDtypeStruct):
                        param.value = jax.ShapeDtypeStruct(
                            param.value.shape, self.quantized_dtype, sharding=param.value.sharding
                        )
                # Static checkpoints will load real scale tensors later, but the
                # placeholders must already satisfy expert sharding shape rules.
                num_experts = self.wi_0.value.shape[0]
                # [E, k, n] layout: wi_0=[E, hidden_size, intermediate_dim],
                #                    wo=[E, intermediate_dim, hidden_size]
                hidden_size = self.wi_0.value.shape[1]
                intermediate_dim = self.wo.value.shape[1]

                # Compute k_blocks for block quant placeholders.
                # weight_block_size = [hf_out_block, hf_in_block] (HF convention).
                # EPMoE quantizes along axis=1 (k/contraction dim).
                block_size_k = _get_block_size_k(
                    hidden_size=hidden_size,
                    intermediate_dim=intermediate_dim,
                    weight_block_size=self.weight_block_size,
                )
                k_blocks_wi = (hidden_size // block_size_k) if block_size_k else 1
                k_blocks_wo = (intermediate_dim // block_size_k) if block_size_k else 1
                wi_scale_sharding = P("expert", None, None, "tensor")
                wo_scale_sharding = P("expert", None, None, None)

                if hasattr(self, "wi_0_scale"):
                    del self.wi_0_scale
                self.wi_0_scale = nnx.Param(
                    jnp.zeros(
                        (num_experts, k_blocks_wi, 1, intermediate_dim),
                        dtype=jnp.float32,
                        out_sharding=wi_scale_sharding,
                    ),
                    out_sharding=wi_scale_sharding,
                )

                if hasattr(self, "wi_1_scale"):
                    del self.wi_1_scale
                self.wi_1_scale = nnx.Param(
                    jnp.zeros(
                        (num_experts, k_blocks_wi, 1, intermediate_dim),
                        dtype=jnp.float32,
                        out_sharding=wi_scale_sharding,
                    ),
                    out_sharding=wi_scale_sharding,
                )

                if hasattr(self, "wo_scale"):
                    del self.wo_scale
                self.wo_scale = nnx.Param(
                    jnp.zeros(
                        (num_experts, k_blocks_wo, 1, hidden_size),
                        dtype=jnp.float32,
                        out_sharding=wo_scale_sharding,
                    ),
                    out_sharding=wo_scale_sharding,
                )
                return

            # Quantize weights along k-dim (axis=1 in [g, k, n] layout)
            # wi_0=[E, hidden_size, intermediate_dim], wo=[E, intermediate_dim, hidden_size]
            hidden_size = self.wi_0.value.shape[1]
            intermediate_dim = self.wo.value.shape[1]
            block_size_k = _get_block_size_k(
                hidden_size=hidden_size,
                intermediate_dim=intermediate_dim,
                weight_block_size=self.weight_block_size,
            )
            w0_value, w0_scale = quantize_tensor(
                self.quantized_dtype,
                self.wi_0.value,
                axis=1,
                block_size=block_size_k,
            )
            w1_value, w1_scale = quantize_tensor(
                self.quantized_dtype,
                self.wi_1.value,
                axis=1,
                block_size=block_size_k,
            )
            wo_value, wo_scale = quantize_tensor(
                self.quantized_dtype,
                self.wo.value,
                axis=1,
                block_size=block_size_k,
            )

            self.wi_0 = nnx.Param(w0_value, out_sharding=P("expert", None, "tensor"))
            self.wi_1 = nnx.Param(w1_value, out_sharding=P("expert", None, "tensor"))
            self.wo = nnx.Param(wo_value, out_sharding=P("expert", "tensor", None))

            if block_size_k is not None:
                # axis=1 quantization on [g, k, n] gives scale [g, k_blocks, n]
                # → expand to [g, k_blocks, 1, n]
                w0_scale = w0_scale[:, :, None, :]
                w1_scale = w1_scale[:, :, None, :]
                wo_scale = wo_scale[:, :, None, :]
            else:
                w0_scale = w0_scale.reshape(w0_scale.shape[0], 1, 1, w0_scale.shape[1])
                w1_scale = w1_scale.reshape(w1_scale.shape[0], 1, 1, w1_scale.shape[1])
                wo_scale = wo_scale.reshape(wo_scale.shape[0], 1, 1, wo_scale.shape[1])

            if hasattr(self, "wi_0_scale"):
                del self.wi_0_scale
            self.wi_0_scale = nnx.Param(
                w0_scale,
                out_sharding=P("expert", None, None, "tensor"),
            )

            if hasattr(self, "wi_1_scale"):
                del self.wi_1_scale
            self.wi_1_scale = nnx.Param(
                w1_scale,
                out_sharding=P("expert", None, None, "tensor"),
            )

            if hasattr(self, "wo_scale"):
                del self.wo_scale
            self.wo_scale = nnx.Param(
                wo_scale,
                out_sharding=P("expert", None, None, None),
            )

    @named_scope
    def __call__(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        *,
        out_sharding: jax.sharding.NamedSharding | None = None,
    ) -> jax.Array:
        total_tokens = (
            hidden_states.shape[0]
            if hidden_states.ndim == 2
            else (hidden_states.shape[0] * hidden_states.shape[1])
        )
        can_scatter_on_expert = (
            _USE_EXPERT_SCATTER
            and self.ep_size > 1
            and self.tp_size == 1
            and self.mesh.shape.get("data", 0) == self.ep_size
            and hidden_states.ndim == 2
            and (total_tokens % self.ep_size == 0)
        )
        if out_sharding is None:
            if can_scatter_on_expert:
                out_sharding = jax.sharding.NamedSharding(self.mesh, P("data", None))
            else:
                out_sharding = jax.sharding.NamedSharding(self.mesh, P(*([None] * hidden_states.ndim)))

        if can_scatter_on_expert and (
            out_sharding.spec[0] == "data"
            or (isinstance(out_sharding.spec[0], tuple) and "data" in out_sharding.spec[0])
        ):
            out_specs = P("expert", None)
            scatter_on_expert = True
            scatter_on_tensor = False
        else:
            out_specs = P(
                *[
                    "tensor" if (s == "tensor" or (isinstance(s, tuple) and "tensor" in s)) else None
                    for s in out_sharding.spec
                ]
            )
            scatter_on_expert = False
            scatter_on_tensor = "tensor" in out_specs

        _pack_routing = (
            topk_weights.dtype == jnp.float32
            and topk_ids.dtype == jnp.int32
            and topk_weights.shape == topk_ids.shape
        )
        if _pack_routing:
            packed_routing = jnp.concatenate(
                [jax.lax.bitcast_convert_type(topk_weights, jnp.int32), topk_ids],
                axis=-1,
            )

        # Run MoE computation on the expert-parallel mesh
        with jax.sharding.use_abstract_mesh(self.updated_mesh):
            hidden_states_reshard = jax.sharding.reshard(hidden_states, P(None))
            if _pack_routing:
                packed_reshard = jax.sharding.reshard(packed_routing, P(None))
                k_top = topk_ids.shape[-1]
                topk_weights_reshard = jax.lax.bitcast_convert_type(
                    packed_reshard[..., :k_top], jnp.float32
                )
                topk_ids_reshard = packed_reshard[..., k_top:]
            else:
                topk_weights_reshard = jax.sharding.reshard(topk_weights, P(None))
                topk_ids_reshard = jax.sharding.reshard(topk_ids, P(None))

            # Normalize scales to GMM's 4D layout [E, k_blocks, 1, out_dim]
            w0_scale = self._normalize_scale_for_gmm(
                self.wi_0_scale.value if self.wi_0_scale is not None else None,
                self.wi_0.value,
                scale_name="wi_0_scale",
            )
            w1_scale = self._normalize_scale_for_gmm(
                self.wi_1_scale.value if self.wi_1_scale is not None else None,
                self.wi_1.value,
                scale_name="wi_1_scale",
            )
            wo_scale = self._normalize_scale_for_gmm(
                self.wo_scale.value if self.wo_scale is not None else None,
                self.wo.value,
                scale_name="wo_scale",
            )

            result = shard_map(
                partial(self._forward, scatter_on_tensor=scatter_on_tensor, scatter_on_expert=scatter_on_expert),
                mesh=self.moe_mesh,
                in_specs=(
                    P(None),
                    P(None),
                    P(None),
                    # weights [g, k, n]
                    P("expert", None, "tensor"),
                    P("expert", None, "tensor"),
                    P("expert", "tensor", None),
                    # scales [g, 1, 1, n]
                    P("expert", None, None, "tensor"),
                    P("expert", None, None, "tensor"),
                    P("expert", None, None, None),
                    # biases [g, 1, n] (unused)
                    P("expert", None, "tensor"),
                    P("expert", None, "tensor"),
                    P("expert", None, None),
                ),
                out_specs=out_specs,
                check_vma=False,
            )(
                hidden_states_reshard,
                topk_weights_reshard,
                topk_ids_reshard,
                self.wi_0.value,
                self.wi_1.value,
                self.wo.value,
                w0_scale,
                w1_scale,
                wo_scale,
                None,
                None,
                None,
            )

        # The shard_map ran under updated_mesh (expert, tensor); land back on
        # the original mesh so downstream ops (residual add, layernorm) see a
        # consistent context.
        return jax.sharding.reshard(result, out_sharding)

    def _forward(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        w0_weights,
        w1_weights,
        wo_weights,
        w0_kernel_scale=None,
        w1_kernel_scale=None,
        wo_kernel_scale=None,
        w0_kernel_bias=None,
        w1_kernel_bias=None,
        wo_kernel_bias=None,
        *,
        scatter_on_tensor: bool = False,
        scatter_on_expert: bool = False,
    ):
        expert_shard_id = jax.lax.axis_index("expert")
        if hidden_states.ndim == 2:
            total_tokens = hidden_states.shape[0]
            batch_size, seq_len = 1, total_tokens
        else:
            batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
            total_tokens = batch_size * seq_len

        inputs_2d, token_indices, sorted_selected_experts, weights, group_sizes = self._permute(
            hidden_states, topk_ids, topk_weights
        )

        group_sizes = group_sizes.astype(jnp.int32)

        group_offset = self._dispatch(group_sizes, expert_shard_id)

        group_offsets = jnp.concatenate(
            [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(group_sizes, dtype=jnp.int32)]
        )
        token_start = group_offsets[group_offset]
        token_end = group_offsets[group_offset + self.experts_per_device]
        use_fused = _USE_FUSED_UNPERMUTE and (total_tokens >= _MOE_FUSED_MIN_TOKENS)

        intermediate_output = self._gmm_compute(
            inputs_2d,
            token_indices,
            group_sizes,
            w0_weights,
            w1_weights,
            wo_weights,
            group_offset,
            w0_kernel_scale,
            w1_kernel_scale,
            wo_kernel_scale,
            w0_kernel_bias,
            w1_kernel_bias,
            wo_kernel_bias,
            token_start=token_start,
            token_end=token_end,
            use_fused=use_fused,
        )

        use_chunked_expert_scatter = (
            scatter_on_expert
            and use_fused
            and _MOE_CHUNK_STAGE > 1
            and (total_tokens % (self.ep_size * _MOE_CHUNK_STAGE) == 0)
        )
        if use_chunked_expert_scatter:
            return self._unpermute_chunked_psum_scatter(
                intermediate_output,
                sorted_selected_experts,
                weights,
                total_tokens=total_tokens,
                token_start=token_start,
                token_end=token_end,
                num_chunks=_MOE_CHUNK_STAGE,
            )

        output = self._unpermute(
            intermediate_output,
            sorted_selected_experts,
            weights,
            batch_size,
            seq_len,
            token_start=token_start,
            token_end=token_end,
            use_fused=use_fused,
        )

        # Reduce on the "tensor" axis. RS (psum_scatter) when caller asked
        # for SP layout on the token dim, AR (psum) otherwise. The matching
        # out_specs is set in __call__ from the same source of truth.
        if self.tp_size > 1:
            if scatter_on_tensor:
                output = jax.lax.psum_scatter(output, "tensor", scatter_dimension=0, tiled=True)
            else:
                output = jax.lax.psum(output, "tensor")
        if self.ep_size > 1:
            if scatter_on_expert:
                output = jax.lax.psum_scatter(output, "expert", scatter_dimension=0, tiled=True)
            else:
                output = self._combine(output)

        return output

    def _unpermute_chunked_psum_scatter(
        self,
        intermediate,
        sorted_selected_experts,
        weights,
        *,
        total_tokens: int,
        token_start,
        token_end,
        num_chunks: int,
    ):
        from sgl_jax.srt.kernels.sparse_core.ragged_gather_reduce_v2 import (
            ragged_gather_reduce,
        )

        expected_tokens = sorted_selected_experts.shape[0]
        actual_tokens = intermediate.shape[0]
        if actual_tokens != expected_tokens:
            if actual_tokens > expected_tokens:
                intermediate = intermediate[:expected_tokens]
            else:
                padding_size = expected_tokens - actual_tokens
                padding = jnp.zeros(
                    (padding_size, intermediate.shape[1]), dtype=intermediate.dtype
                )
                intermediate = jnp.concatenate([intermediate, padding], axis=0)

        argsort_indices = jnp.argsort(sorted_selected_experts).astype(jnp.int32)
        flat_weights = jnp.reshape(weights, (-1,))
        if self.ep_size > 1 and token_start is not None and token_end is not None:
            valid_rows = (argsort_indices >= token_start) & (
                argsort_indices < token_end
            )
        else:
            valid_rows = jnp.ones((expected_tokens,), dtype=jnp.bool_)

        top_k = self.num_experts_per_tok
        idx_by_chunk = (
            argsort_indices.reshape(self.ep_size, num_chunks, -1, top_k)
            .transpose(1, 0, 2, 3)
            .reshape(num_chunks, -1)
        )
        w_by_chunk = (
            flat_weights.reshape(self.ep_size, num_chunks, -1, top_k)
            .transpose(1, 0, 2, 3)
            .reshape(num_chunks, -1)
        )
        val_by_chunk = (
            valid_rows.reshape(self.ep_size, num_chunks, -1, top_k)
            .transpose(1, 0, 2, 3)
            .reshape(num_chunks, -1)
        )

        pieces = []
        for c in range(num_chunks):
            chunk_out = ragged_gather_reduce(
                intermediate,
                idx_by_chunk[c],
                w_by_chunk[c],
                val_by_chunk[c],
                top_k,
            ).astype(self.dtype)
            piece = jax.lax.psum_scatter(
                chunk_out, "expert", scatter_dimension=0, tiled=True
            )
            pieces.append(piece)

        return jnp.concatenate(pieces, axis=0)

    def _gmm_compute(
        self,
        inputs_2d,
        token_indices,
        group_sizes,
        w0_kernel,
        w1_kernel,
        wo_kernel,
        group_offset,
        w0_kernel_scale=None,
        w1_kernel_scale=None,
        wo_kernel_scale=None,
        w0_kernel_bias=None,
        w1_kernel_bias=None,
        wo_kernel_bias=None,
        *,
        token_start=None,
        token_end=None,
        use_fused: bool = False,
    ):
        if token_indices.shape[0] == 0:
            return jnp.zeros((0, wo_kernel.shape[-1]), dtype=inputs_2d.dtype)

        # indexed_gmm: gather sorted_inputs here instead of in _permute,
        # so XLA can fuse the gather with the matmul and avoid materializing
        # the full [M*top_k, D] sorted_inputs tensor at peak memory.
        pre_gather_q = getattr(self, "pre_gather_quant_dtype", None)
        if pre_gather_q is not None:
            x_q, x_scale = quantize_tensor_simple(inputs_2d, pre_gather_q, dim=-1)
            x = x_q[token_indices]
            x_scale = x_scale[token_indices]
            x = (x.astype(jnp.float32) * x_scale).astype(self.dtype)
        elif (
            use_fused
            and _USE_RAGGED_GATHER
            and self.ep_size > 1
            and token_start is not None
            and token_end is not None
        ):
            from sgl_jax.srt.kernels.sparse_core.ragged_gather_v2 import (
                ragged_gather_v2,
            )

            x = ragged_gather_v2(
                inputs_2d.astype(self.dtype),
                token_indices,
                token_start,
                token_end,
            )
        else:
            x = inputs_2d[token_indices].astype(self.dtype)

        # NOTE: do NOT pad LHS / bump group_sizes here. The megablox backend
        # ``gmm`` (sgl_jax/srt/kernels/gmm/megablox_gmm_backend.py:67-73)
        # already pads ``lhs`` to its required alignment (32 for v2, 128 for
        # v1), bumps ``group_sizes[-1]`` accordingly, and slices the output
        # back to the original ``m`` afterwards. An outer pre-pad is at best
        # redundant; in practice the previous workaround pre-padded to a
        # hard-coded ``128`` which forced v2 (alignment=32) into a 4x larger
        # tile, hit a kernel auto-tiler edge case at decode bs=8 / top_k=8
        # (m=64 -> 128) and triggered an on-device SparseCore halt.
        group_sizes = group_sizes.astype(jnp.int32)
        act_q_dtype = self.activation_quantized_dtype

        gmm_kwargs = dict(
            group_sizes=group_sizes,
            preferred_element_type=self.dtype,
            group_offset=group_offset,
            maybe_quantize_lhs=act_q_dtype is not None,
            acc_dtype=jnp.float32,
        )

        # === GEMM1: x @ w0 and x @ w1 ===
        layer_w0 = gmm(
            lhs=x,
            rhs=w0_kernel,
            rhs_scale=w0_kernel_scale,
            rhs_bias=w0_kernel_bias,
            zero_initialize=False,
            activation_quantized_dtype=act_q_dtype,
            **gmm_kwargs,
        )
        layer_w1 = gmm(
            lhs=x,
            rhs=w1_kernel,
            rhs_scale=w1_kernel_scale,
            rhs_bias=w1_kernel_bias,
            zero_initialize=False,
            activation_quantized_dtype=act_q_dtype,
            **gmm_kwargs,
        )

        # === Activation ===
        if (
            self.activation == "silu"
            and use_fused
            and self.ep_size > 1
            and token_start is not None
            and token_end is not None
            and layer_w0.ndim == 2
            and layer_w0.shape[0] >= 4096
            and layer_w0.shape[0] % 1024 == 0
            and layer_w0.shape[1] % 128 == 0
        ):
            intermediate_layer = _ranged_swiglu(
                layer_w0, layer_w1, token_start, token_end
            )
        else:
            if self.activation == "silu":
                layer_act = jax.nn.silu(layer_w0)
            elif self.activation == "gelu":
                layer_act = jax.nn.gelu(layer_w0)
            else:
                raise ValueError(f"Unsupported activation function {self.activation}")
            intermediate_layer = jnp.multiply(layer_act, layer_w1)

        # === GEMM2: intermediate @ wo ===
        # When use_fused is True and ep_size > 1, ragged_gather_reduce_v2 masks
        # out all rows outside [token_start, token_end) via valid_rows, so
        # zero_initialize=False avoids zero-filling 31/32 of intermediate_output.
        return gmm(
            lhs=intermediate_layer,
            rhs=wo_kernel,
            rhs_scale=wo_kernel_scale,
            rhs_bias=wo_kernel_bias,
            zero_initialize=not (use_fused and self.ep_size > 1),
            activation_quantized_dtype=act_q_dtype,
            **gmm_kwargs,
        )

    def _dispatch(self, group_sizes, expert_shard_id):
        if self.ep_size <= 1:
            return jnp.array(0, dtype=jnp.int32)
        group_offset = jnp.array(expert_shard_id * self.experts_per_device, dtype=jnp.int32)
        return group_offset

    def _get_all_to_all_params(
        self,
        tokens_group: jax.Array,
        shard_id: jax.Array,
        start_idx: jax.Array,
        *,
        ep_size: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        input_offsets = jnp.full(ep_size, start_idx, dtype=tokens_group.dtype)
        send_sizes = jnp.repeat(tokens_group[shard_id], ep_size)
        output_offset = jnp.concatenate(
            (jnp.array([0], dtype=tokens_group.dtype), jnp.cumsum(tokens_group[:-1]))
        )[shard_id]
        output_offsets = jnp.repeat(output_offset, ep_size)
        recv_sizes = tokens_group

        return input_offsets, send_sizes, output_offsets, recv_sizes

    def _combine(self, data):
        return jax.lax.psum(data, "expert")

    def _permute(self, inputs, top_k_indices, top_k_weights):
        inputs_shape = inputs.shape

        if len(inputs_shape) == 2:
            inputs_2d = inputs
            bsz_times_seq_len = inputs_shape[0]
        else:
            bsz_times_seq_len = inputs_shape[0] * inputs_shape[1]
            inputs_2d = jnp.reshape(inputs, (bsz_times_seq_len, inputs_shape[-1]))

        del bsz_times_seq_len

        flatten_selected_experts = jnp.ravel(top_k_indices)
        sorted_selected_experts = jnp.argsort(flatten_selected_experts, stable=True)
        # token_indices: maps each sorted position to the original token index.
        # Pass to _gmm_compute so the gather happens there (indexed_gmm pattern),
        # avoiding a full [M*top_k, D] materialization in _permute.
        token_indices = sorted_selected_experts // self.num_experts_per_tok

        group_sizes = jnp.bincount(flatten_selected_experts, length=self.num_experts)

        return (
            inputs_2d,
            token_indices,
            sorted_selected_experts,
            top_k_weights,
            group_sizes,
        )

    def _unpermute(
        self,
        intermediate,
        sorted_selected_experts,
        weights,
        batch_size,
        seq_len,
        *,
        token_start=None,
        token_end=None,
        use_fused: bool = False,
    ):
        expected_tokens = sorted_selected_experts.shape[0]
        actual_tokens = intermediate.shape[0]

        if actual_tokens != expected_tokens:
            if actual_tokens > expected_tokens:
                intermediate = intermediate[:expected_tokens]
            else:
                padding_size = expected_tokens - actual_tokens
                padding = jnp.zeros((padding_size, intermediate.shape[1]), dtype=intermediate.dtype)
                intermediate = jnp.concatenate([intermediate, padding], axis=0)

        argsort_indices = jnp.argsort(sorted_selected_experts).astype(jnp.int32)

        total_tokens = weights.shape[0] * weights.shape[1] // self.num_experts_per_tok

        if use_fused:
            # Fused path: one SparseCore kernel does gather -> weight multiply ->
            # mask -> sum over top_k, emitting [total_tokens, hidden] directly.
            # The [expected_tokens, hidden] intermediate is never materialised,
            # which is an ``num_experts_per_tok``-fold cut in bytes written.
            from sgl_jax.srt.kernels.sparse_core.ragged_gather_reduce_v2 import (
                ragged_gather_reduce,
            )

            # Row i of the unsorted intermediate is assignment i = token * top_k
            # + k, so the flattened weights line up elementwise with it.
            flat_weights = jnp.reshape(weights, (-1,))
            # In EPMoE, only rows in [token_start, token_end) were computed by
            # this rank's experts; masking out the other 31/32 rows lets
            # ragged_gather_reduce_v2 skip 31/32 of SparseCore HBM DMA gathers!
            if self.ep_size > 1 and token_start is not None and token_end is not None:
                valid_rows = (argsort_indices >= token_start) & (
                    argsort_indices < token_end
                )
            else:
                valid_rows = jnp.ones((expected_tokens,), dtype=jnp.bool_)

            if _MOE_FUSED_DEBUG:
                print(
                    f"[FUSED-UNPERMUTE] layer={getattr(self, 'layer_id', '?')} "
                    f"x={tuple(intermediate.shape)}/{intermediate.dtype} "
                    f"idx={tuple(argsort_indices.shape)} "
                    f"w={tuple(flat_weights.shape)}/{flat_weights.dtype} "
                    f"top_k={self.num_experts_per_tok} "
                    f"stage={_MOE_CHUNK_STAGE} "
                    f"-> out=({total_tokens}, {intermediate.shape[-1]})",
                    flush=True,
                )

            top_k = self.num_experts_per_tok
            if _MOE_CHUNK_STAGE > 1 and total_tokens > _MOE_CHUNK_STAGE:
                chunk_tokens = -(-total_tokens // _MOE_CHUNK_STAGE)  # ceil
                pieces = []
                for start_tok in range(0, total_tokens, chunk_tokens):
                    end_tok = min(total_tokens, start_tok + chunk_tokens)
                    lo, hi = start_tok * top_k, end_tok * top_k
                    if _MOE_FUSED_DEBUG:
                        print(
                            f"[FUSED-UNPERMUTE]   chunk tokens[{start_tok}:{end_tok}] "
                            f"assignments[{lo}:{hi}] size={hi - lo}",
                            flush=True,
                        )
                    pieces.append(
                        ragged_gather_reduce(
                            intermediate,
                            argsort_indices[lo:hi],
                            flat_weights[lo:hi],
                            valid_rows[lo:hi],
                            top_k,
                        )
                    )
                output = (
                    jnp.concatenate(pieces, axis=0) if len(pieces) > 1 else pieces[0]
                )
            else:
                output = ragged_gather_reduce(
                    intermediate,
                    argsort_indices,
                    flat_weights,
                    valid_rows,
                    top_k,
                )
        else:
            unsort_intermediate = jnp.take(intermediate, indices=argsort_indices, axis=0)

            reshaped_weights = jnp.reshape(weights, (total_tokens, self.num_experts_per_tok))
            reshaped_intermediate = jnp.reshape(
                unsort_intermediate,
                (total_tokens, self.num_experts_per_tok, -1),
            )

            intermediate_fp32 = reshaped_intermediate.astype(jnp.float32)
            weights_fp32 = reshaped_weights.astype(jnp.float32)

            output = jnp.einsum(
                "BKE,BK -> BE",
                intermediate_fp32,
                weights_fp32,
            )

        if len(weights.shape) == 2:
            final_output = output.astype(self.dtype)
        else:
            final_output = output.reshape(batch_size, seq_len, -1).astype(self.dtype)

        return final_output


# create_moe_weights_mapping is utility function to generate weight mapping for MOE layers
def create_moe_weights_mapping(
    prefix: str,
    target_prefix: str,
    num_experts: int,  # num logical experts
    expert_type_names: tuple[str, str, str] = (
        "gate_proj",
        "up_proj",
        "down_proj",
    ),  # expert source names [gate, up, down]
    expert_concat_axis_map: dict[
        str, int
    ] = None,  # Map from source weight name to its concatenation axis (default is None)
    moe_backend: str = "epmoe",
    moe_path: str = "mlp",
    source_expert_pattern: str = "experts.{i}",
    physical_to_logical_map=None,  # np.ndarray shape (num_physical,) or None
) -> dict:
    """Generate a unified mapping dictionary for MoE layer expert weights."""
    if moe_backend == "epmoe":
        expert_type_map = {
            expert_type_names[0]: "wi_0",
            expert_type_names[1]: "wi_1",
            expert_type_names[2]: "wo",
        }
    elif moe_backend in ("fused", "fused_v2"):
        expert_type_map = {
            expert_type_names[0]: "w1",
            expert_type_names[1]: "w3",
            expert_type_names[2]: "w2",
        }
    else:
        raise ValueError(f"Unsupported MoE backend: {moe_backend}")

    if expert_concat_axis_map is None:
        expert_concat_axis_map = {}

    mappings = {}
    for source_name, target_name in expert_type_map.items():
        # Target path for JAX model parameters (matching EPMoE internal variables)
        target_path_base = f"{target_prefix}.{moe_path}.{target_name}"

        # Source weight paths for logical experts only
        expert_keys = [
            f"{prefix}.{moe_path}.{source_expert_pattern.format(i=i)}.{source_name}.weight"
            for i in range(num_experts)
        ]

        if moe_backend == "epmoe":
            # Weights are transposed from HF [n, k] to [k, n], stacked to [g, k, n].
            # wi_0/wi_1: [g, hidden_size, intermediate_dim] -> P("expert", None, "tensor")
            # wo:        [g, intermediate_dim, hidden_size] -> P("expert", "tensor", None)
            sharding = (
                ("expert", "tensor", None) if target_name == "wo" else ("expert", None, "tensor")
            )
            transpose = True
        elif moe_backend in ("fused", "fused_v2"):
            # Fused MoE kernel shards experts across the full EP mesh, i.e. the
            # product of ("data", "tensor"). Shard expert dim (axis=0) across
            # both mesh axes so each device owns a disjoint expert slice.
            sharding = (("data", "tensor"), None, None)
            transpose = True
        else:
            raise ValueError(f"Unsupported MoE backend: {moe_backend}")

        concat_axis = expert_concat_axis_map.get(source_name)

        # Use  prefix to indicate aggregated MoE weight loading
        mappings[f"{target_path_base}"] = WeightSpec(
            target_path=target_path_base,
            sources=tuple(expert_keys),
            sharding=sharding,
            transpose=transpose,
            concat_axis=concat_axis,
            physical_to_logical_map=physical_to_logical_map,
        )

    return mappings
