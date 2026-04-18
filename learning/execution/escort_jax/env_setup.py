from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from crazy_rl.multi_agent.jax.escort import Escort
from crazy_rl.utils.jax_wrappers import AddIDToObs, AutoReset, ClipActions, LogWrapper, NormalizeObservation, NormalizeVecReward, VecEnv


def make_init_positions(num_drones: int, radius: float = 0.85, z: float = 1.0) -> jnp.ndarray:
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, num_drones, endpoint=False)
    x = radius * jnp.cos(angles)
    y = radius * jnp.sin(angles)
    z_arr = jnp.ones_like(x) * z
    return jnp.stack([x, y, z_arr], axis=1)


def make_escort_env(
    num_drones: int,
    size: int,
    gamma: float,
    init_target_location: jnp.ndarray,
    final_target_location: jnp.ndarray,
    num_intermediate_points: int,
    normalize_reward: bool,
    target_speed_multiplier: float = 1.0,
    random_horizontal_path: bool = False,
    target_height_min: float = 1.0,
    target_height_max: float = 1.0,
    target_segment_length: float = 5.0,
    target_curvature_scale: float = 0.35,
    smoothness_coef: float = 0.0,
    action_l2_coef: float = 0.0,
    neighbor_symmetry_coef: float = 0.0,
    angular_uniformity_coef: float = 0.0,
    heading_alignment_coef: float = 0.0,
    init_flying_pos: Optional[jnp.ndarray] = None,
):
    init_flying_pos = make_init_positions(num_drones) if init_flying_pos is None else init_flying_pos

    def _out_of_bounds(p: jnp.ndarray) -> bool:
        xy_oob = bool(jnp.any(jnp.abs(p[:2]) > float(size)))
        z_oob = bool((p[2] < 0.0) | (p[2] > float(size)))
        return xy_oob or z_oob

    if _out_of_bounds(init_target_location) or _out_of_bounds(final_target_location):
        raise ValueError(
            "Target coordinates are out of map bounds. Expected x,y in [-size, size] and z in [0, size]. "
            f"Got init={init_target_location.tolist()}, final={final_target_location.tolist()}, size={size}."
        )

    env = Escort(
        num_drones=num_drones,
        init_flying_pos=init_flying_pos,
        init_target_location=init_target_location,
        final_target_location=final_target_location,
        num_intermediate_points=num_intermediate_points,
        target_speed_multiplier=target_speed_multiplier,
        random_horizontal_path=random_horizontal_path,
        target_height_min=target_height_min,
        target_height_max=target_height_max,
        target_segment_length=target_segment_length,
        target_curvature_scale=target_curvature_scale,
        smoothness_coef=smoothness_coef,
        action_l2_coef=action_l2_coef,
        neighbor_symmetry_coef=neighbor_symmetry_coef,
        angular_uniformity_coef=angular_uniformity_coef,
        heading_alignment_coef=heading_alignment_coef,
        multi_obj=False,
        size=size,
    )
    env = ClipActions(env)
    env = NormalizeObservation(env)
    env = AddIDToObs(env, num_drones)
    env = LogWrapper(env)
    env = AutoReset(env)
    env = VecEnv(env)
    if normalize_reward:
        env = NormalizeVecReward(env, gamma)
    return env
