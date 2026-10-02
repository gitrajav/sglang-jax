from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

RELAY_STATE_SPEC = P("data", None, None)
RELAY_ID_SPEC = P("data", None)


class SpecRelayBuffers(NamedTuple):
    topk_index: jax.Array
    hidden_states: jax.Array
    verified_id: jax.Array
    new_seq_lens: jax.Array


class DFlashRelayBuffers(NamedTuple):
    verified_id: jax.Array
    new_seq_lens: jax.Array


def create_spec_relay_buffers(
    mesh,
    req_to_token_pool,
    *,
    dp_size: int,
    num_steps: int,
    hidden_size: int,
    hidden_dtype,
) -> SpecRelayBuffers:
    """Create DP-local req-indexed buffers for cross-batch draft state relay."""
    capacity = int(req_to_token_pool.req_to_token.shape[0])
    token_sharding = NamedSharding(mesh, RELAY_STATE_SPEC)
    hidden_sharding = NamedSharding(mesh, RELAY_STATE_SPEC)
    id_sharding = NamedSharding(mesh, RELAY_ID_SPEC)
    return SpecRelayBuffers(
        topk_index=jax.device_put(
            jnp.zeros((dp_size, capacity, num_steps), dtype=jnp.int32),
            token_sharding,
        ),
        hidden_states=jax.device_put(
            jnp.zeros((dp_size, capacity, hidden_size), dtype=hidden_dtype),
            hidden_sharding,
        ),
        verified_id=jax.device_put(
            jnp.zeros((dp_size, capacity), dtype=jnp.int32),
            id_sharding,
        ),
        new_seq_lens=jax.device_put(
            jnp.zeros((dp_size, capacity), dtype=jnp.int32),
            id_sharding,
        ),
    )


def create_dflash_relay_buffers(
    mesh,
    req_to_token_pool,
    *,
    dp_size: int,
) -> DFlashRelayBuffers:
    """Create the minimal req-indexed state needed by DFlash overlap."""
    capacity = int(req_to_token_pool.req_to_token.shape[0])
    sharding = NamedSharding(mesh, RELAY_ID_SPEC)
    shape = (dp_size, capacity)
    return DFlashRelayBuffers(
        verified_id=jax.device_put(jnp.zeros(shape, dtype=jnp.int32), sharding),
        new_seq_lens=jax.device_put(jnp.zeros(shape, dtype=jnp.int32), sharding),
    )


def update_spec_relay_buffers(
    buffers: SpecRelayBuffers,
    future_indices,
    valid_mask,
    topk_index,
    hidden_states,
    verified_id,
    new_seq_lens,
    *,
    dp_size: int,
) -> SpecRelayBuffers:
    """Write DP-padded draft state into relay buffers without touching padded rows."""
    per_dp_bs = future_indices.shape[0] // dp_size
    flat_sharding = jax.typeof(future_indices).sharding
    if (
        isinstance(flat_sharding, NamedSharding)
        and not flat_sharding.mesh.empty
        and "data" in flat_sharding.mesh.axis_names
        and flat_sharding.mesh.shape["data"] == dp_size
    ):
        mesh = flat_sharding.mesh
        d1_sharding = NamedSharding(mesh, P("data"))
        d2_sharding = NamedSharding(mesh, P("data", None))
        if jax.typeof(valid_mask).sharding != d1_sharding:
            valid_mask = jax.sharding.reshard(valid_mask, d1_sharding)
        if jax.typeof(verified_id).sharding != d1_sharding:
            verified_id = jax.sharding.reshard(verified_id, d1_sharding)
        if jax.typeof(new_seq_lens).sharding != d1_sharding:
            new_seq_lens = jax.sharding.reshard(new_seq_lens, d1_sharding)
        if jax.typeof(topk_index).sharding != d2_sharding:
            topk_index = jax.sharding.reshard(topk_index, d2_sharding)
        if jax.typeof(hidden_states).sharding != d2_sharding:
            hidden_states = jax.sharding.reshard(hidden_states, d2_sharding)

        def _local_update(tk_b, hs_b, vid_b, nsl_b, f_idx, v_mask, tk_u, hs_u, vid_u, nsl_u):
            idx = f_idx.reshape((per_dp_bs,))
            val = v_mask.reshape((per_dp_bs,))
            s_idx = jnp.where(val, idx, jnp.full_like(idx, tk_b.shape[1]))
            return SpecRelayBuffers(
                topk_index=tk_b.at[0, s_idx].set(
                    tk_u.reshape((per_dp_bs,) + tk_u.shape[1:]), mode="drop"
                ),
                hidden_states=hs_b.at[0, s_idx].set(
                    hs_u.reshape((per_dp_bs,) + hs_u.shape[1:]), mode="drop"
                ),
                verified_id=vid_b.at[0, s_idx].set(vid_u.reshape((per_dp_bs,)), mode="drop"),
                new_seq_lens=nsl_b.at[0, s_idx].set(nsl_u.reshape((per_dp_bs,)), mode="drop"),
            )

        return jax.shard_map(
            _local_update,
            mesh=mesh,
            in_specs=(
                RELAY_STATE_SPEC,
                RELAY_STATE_SPEC,
                RELAY_ID_SPEC,
                RELAY_ID_SPEC,
                P("data"),
                P("data"),
                P("data", None),
                P("data", None),
                P("data"),
                P("data"),
            ),
            out_specs=SpecRelayBuffers(
                topk_index=RELAY_STATE_SPEC,
                hidden_states=RELAY_STATE_SPEC,
                verified_id=RELAY_ID_SPEC,
                new_seq_lens=RELAY_ID_SPEC,
            ),
            check_vma=False,
        )(
            buffers.topk_index,
            buffers.hidden_states,
            buffers.verified_id,
            buffers.new_seq_lens,
            future_indices,
            valid_mask,
            topk_index,
            hidden_states,
            verified_id,
            new_seq_lens,
        )

    indices = future_indices.reshape((dp_size, per_dp_bs))
    valid = valid_mask.reshape((dp_size, per_dp_bs))
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    scatter_indices = jnp.where(
        valid,
        indices,
        jnp.full_like(indices, buffers.topk_index.shape[1]),
    )

    topk_index = topk_index.reshape((dp_size, per_dp_bs) + topk_index.shape[1:])
    hidden_states = hidden_states.reshape((dp_size, per_dp_bs) + hidden_states.shape[1:])
    verified_id = verified_id.reshape((dp_size, per_dp_bs))
    new_seq_lens = new_seq_lens.reshape((dp_size, per_dp_bs))

    return SpecRelayBuffers(
        topk_index=buffers.topk_index.at[dp_indices, scatter_indices].set(
            topk_index,
            mode="drop",
            out_sharding=RELAY_STATE_SPEC,
        ),
        hidden_states=buffers.hidden_states.at[dp_indices, scatter_indices].set(
            hidden_states,
            mode="drop",
            out_sharding=RELAY_STATE_SPEC,
        ),
        verified_id=buffers.verified_id.at[dp_indices, scatter_indices].set(
            verified_id,
            mode="drop",
            out_sharding=RELAY_ID_SPEC,
        ),
        new_seq_lens=buffers.new_seq_lens.at[dp_indices, scatter_indices].set(
            new_seq_lens,
            mode="drop",
            out_sharding=RELAY_ID_SPEC,
        ),
    )


def update_dflash_relay_buffers(
    buffers: DFlashRelayBuffers,
    future_indices,
    valid_mask,
    verified_id,
    new_seq_lens,
    *,
    dp_size: int,
) -> DFlashRelayBuffers:
    """Publish one DP-padded DFlash round without writing padded slots."""
    per_dp_bs = future_indices.shape[0] // dp_size
    indices = future_indices.reshape((dp_size, per_dp_bs))
    valid = valid_mask.reshape((dp_size, per_dp_bs))
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    scatter_indices = jnp.where(
        valid,
        indices,
        jnp.full_like(indices, buffers.verified_id.shape[1]),
    )
    verified_id = verified_id.reshape((dp_size, per_dp_bs))
    new_seq_lens = new_seq_lens.reshape((dp_size, per_dp_bs))
    return DFlashRelayBuffers(
        verified_id=buffers.verified_id.at[dp_indices, scatter_indices].set(
            verified_id,
            mode="drop",
            out_sharding=RELAY_ID_SPEC,
        ),
        new_seq_lens=buffers.new_seq_lens.at[dp_indices, scatter_indices].set(
            new_seq_lens,
            mode="drop",
            out_sharding=RELAY_ID_SPEC,
        ),
    )


def gather_spec_relay_buffers(
    buffers: SpecRelayBuffers,
    future_indices,
    *,
    dp_size: int,
):
    """Gather DP-padded draft state for the next batch."""
    per_dp_bs = future_indices.shape[0] // dp_size
    flat_sharding = jax.typeof(future_indices).sharding
    if (
        isinstance(flat_sharding, NamedSharding)
        and not flat_sharding.mesh.empty
        and "data" in flat_sharding.mesh.axis_names
        and flat_sharding.mesh.shape["data"] == dp_size
    ):
        mesh = flat_sharding.mesh

        def _local_gather(tk_b, hs_b, vid_b, nsl_b, f_idx):
            idx = f_idx.reshape((per_dp_bs,))
            return (
                tk_b[0, idx],
                hs_b[0, idx],
                vid_b[0, idx],
                nsl_b[0, idx],
            )

        return jax.shard_map(
            _local_gather,
            mesh=mesh,
            in_specs=(
                RELAY_STATE_SPEC,
                RELAY_STATE_SPEC,
                RELAY_ID_SPEC,
                RELAY_ID_SPEC,
                P("data"),
            ),
            out_specs=(
                P("data", None),
                P("data", None),
                P("data"),
                P("data"),
            ),
            check_vma=False,
        )(
            buffers.topk_index,
            buffers.hidden_states,
            buffers.verified_id,
            buffers.new_seq_lens,
            future_indices,
        )

    indices = future_indices.reshape((dp_size, per_dp_bs))
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    topk_index = (
        buffers.topk_index.at[dp_indices, indices]
        .get(out_sharding=RELAY_STATE_SPEC)
        .reshape(future_indices.shape + buffers.topk_index.shape[2:])
    )
    hidden_states = (
        buffers.hidden_states.at[dp_indices, indices]
        .get(out_sharding=RELAY_STATE_SPEC)
        .reshape(future_indices.shape + buffers.hidden_states.shape[2:])
    )
    verified_id = (
        buffers.verified_id.at[dp_indices, indices]
        .get(out_sharding=RELAY_ID_SPEC)
        .reshape(future_indices.shape)
    )
    new_seq_lens = (
        buffers.new_seq_lens.at[dp_indices, indices]
        .get(out_sharding=RELAY_ID_SPEC)
        .reshape(future_indices.shape)
    )
    if isinstance(flat_sharding, NamedSharding) and not flat_sharding.mesh.empty:
        state_sharding = NamedSharding(flat_sharding.mesh, P("data", None))
        topk_index = jax.sharding.reshard(topk_index, state_sharding)
        hidden_states = jax.sharding.reshard(hidden_states, state_sharding)
        verified_id = jax.sharding.reshard(verified_id, flat_sharding)
        new_seq_lens = jax.sharding.reshard(new_seq_lens, flat_sharding)
    return topk_index, hidden_states, verified_id, new_seq_lens


def gather_dflash_relay_buffers(
    buffers: DFlashRelayBuffers,
    future_indices,
    *,
    dp_size: int,
):
    """Gather the DFlash seed token and logical length for the next round."""
    per_dp_bs = future_indices.shape[0] // dp_size
    indices = future_indices.reshape((dp_size, per_dp_bs))
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    verified_id = (
        buffers.verified_id.at[dp_indices, indices]
        .get(out_sharding=RELAY_ID_SPEC)
        .reshape(future_indices.shape)
    )
    new_seq_lens = (
        buffers.new_seq_lens.at[dp_indices, indices]
        .get(out_sharding=RELAY_ID_SPEC)
        .reshape(future_indices.shape)
    )
    flat_sharding = jax.typeof(future_indices).sharding
    if isinstance(flat_sharding, NamedSharding) and not flat_sharding.mesh.empty:
        verified_id = jax.sharding.reshard(verified_id, flat_sharding)
        new_seq_lens = jax.sharding.reshard(new_seq_lens, flat_sharding)
    return verified_id, new_seq_lens


def make_dp_valid_mask(real_bs_per_dp, *, total_bs: int, per_dp_bs: int) -> np.ndarray:
    mask = np.zeros((total_bs,), dtype=np.bool_)
    for dp_rank, real_bs in enumerate(real_bs_per_dp):
        if real_bs:
            start = dp_rank * per_dp_bs
            mask[start : start + int(real_bs)] = True
    return mask
