"""Escort environment for Crazyflie 2. Each agent is supposed to learn to surround a common target point moving to one point to another."""

from functools import partial
from typing import Tuple
from typing_extensions import override

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jax import jit, random

from crazy_rl.multi_agent.jax.base_parallel_env import (
    CLOSENESS_THRESHOLD,
    BaseParallelEnv,
    State,
)
from crazy_rl.utils.jax_spaces import Box, Space
from crazy_rl.utils.jax_wrappers import AutoReset, VecEnv


@jdc.pytree_dataclass
class State(State):
    """State of the environment containing the modifiable variables."""

    agents_locations: jnp.ndarray  # a 2D array containing x,y,z coordinates of each agent, indexed from 0.
    timestep: int  # represents the number of steps already done in the game
    target_location: jnp.ndarray  # 2D array containing x,y,z coordinates of the common target
    prev_agent_locations: jnp.ndarray  # 2D array containing x,y,z coordinates of each agent at last timestep
    prev_target_locations: jnp.ndarray  # 2D array containing x,y,z coordinates of the target of each agent at last timestep
    prev_actions: jnp.ndarray  # 2D array containing x,y,z actions from previous timestep
    current_actions: jnp.ndarray  # 2D array containing x,y,z actions from current timestep
    target_path: jnp.ndarray  # 2D array containing the sampled target path for the current episode


class Escort(BaseParallelEnv):
    """A Parallel Environment where drone learn how to surround a moving target going straight to one point to another."""

    def __init__(
        self,
        num_drones: int,
        init_flying_pos: jnp.ndarray,
        init_target_location: jnp.ndarray,
        final_target_location: jnp.ndarray,
        num_intermediate_points: int = 20,
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
        multi_obj: bool = False,
        size: int = 2,
    ):
        """Escort environment for Crazyflies 2.

        Args:
            num_drones: Number of drones
            init_flying_pos: Array of initial positions of the drones when they are flying
            init_target_location: Array of the initial position of the moving target
            final_target_location: Array of the final position of the moving target
            num_intermediate_points: Number of intermediate points in the target trajectory
            target_speed_multiplier: Constant speed multiplier for the target path; values > 1.0 make the target move faster.
            random_horizontal_path: If True, sample a random horizontal target segment each episode.
            target_height_min: Lower bound for random target height (meters).
            target_height_max: Upper bound for random target height (meters).
            target_segment_length: Desired segment length (meters) for random horizontal target segments.
            target_curvature_scale: Relative lateral curvature used for random horizontal target paths.
            smoothness_coef: Penalty weight for action change between consecutive steps.
            action_l2_coef: Penalty weight for action magnitude.
            neighbor_symmetry_coef: Penalty weight for spread of nearest-neighbor distances.
            angular_uniformity_coef: Penalty weight for angular spacing irregularity around target.
            heading_alignment_coef: Reward weight for matching swarm heading with target heading.
            multi_obj: Whether to return a multi-objective reward
            size: Size of the map in meters
        """
        self.num_drones = num_drones

        self._target_location = init_target_location  # unique target location for all agents
        self._init_target_location = init_target_location
        self._final_target_location = final_target_location

        self._init_flying_pos = init_flying_pos
        self.multi_obj = multi_obj
        self.target_speed_multiplier = float(target_speed_multiplier)
        self.random_horizontal_path = bool(random_horizontal_path)
        self.target_height_min = float(target_height_min)
        self.target_height_max = float(target_height_max)
        self.target_segment_length = float(target_segment_length)
        self.target_curvature_scale = float(target_curvature_scale)
        self.smoothness_coef = float(smoothness_coef)
        self.action_l2_coef = float(action_l2_coef)
        self.neighbor_symmetry_coef = float(neighbor_symmetry_coef)
        self.angular_uniformity_coef = float(angular_uniformity_coef)
        self.heading_alignment_coef = float(heading_alignment_coef)
        self.formation_radius = 0.9
        # Episode always lasts 200 timesteps, so target path has exactly 200 points.
        self.num_ref_points = 200

        self.size = size
        self.target_path = self._build_target_path()

    @partial(jit, static_argnums=(0,))
    def _formation_components(self, state: State) -> Tuple[jnp.ndarray, ...]:
        target = state.target_location[0]
        centroid = jnp.mean(state.agents_locations, axis=0)
        prev_centroid = jnp.mean(state.prev_agent_locations, axis=0)
        prev_target = state.prev_target_locations[0]

        centroid_progress = jnp.linalg.norm(prev_centroid - prev_target) - jnp.linalg.norm(centroid - target)
        radii = jnp.linalg.norm(state.agents_locations - target, axis=1)
        mean_radius = jnp.mean(radii)
        radius_error = jnp.abs(mean_radius - self.formation_radius)
        radius_spread = jnp.std(radii)

        pairwise_distances = jnp.linalg.norm(
            state.agents_locations[:, None, :] - state.agents_locations[None, :, :], axis=-1
        )
        non_diagonal = pairwise_distances + jnp.eye(self.num_drones) * 1e6
        nearest_neighbor_distances = jnp.min(non_diagonal, axis=1)
        neighbor_distance_spread = jnp.std(nearest_neighbor_distances)
        collision_penalty = jnp.where(
            jnp.any(jnp.logical_and(pairwise_distances > 0.001, pairwise_distances < CLOSENESS_THRESHOLD)),
            -10.0,
            0.0,
        )

        action_change = jnp.linalg.norm(state.current_actions - state.prev_actions, axis=1)
        reward_smooth = -self.smoothness_coef * action_change
        action_magnitude = jnp.linalg.norm(state.current_actions, axis=1)
        reward_action_l2 = -self.action_l2_coef * action_magnitude
        reward_smooth_mean = jnp.mean(reward_smooth)
        reward_action_l2_mean = jnp.mean(reward_action_l2)

        # Encourage uniform angular spacing around the target without assigning fixed drone slots.
        rel_xy = state.agents_locations[:, :2] - target[:2]
        angles = jnp.arctan2(rel_xy[:, 1], rel_xy[:, 0])
        sorted_angles = jnp.sort(angles)
        wrapped_next = jnp.concatenate([sorted_angles[1:], sorted_angles[:1] + 2.0 * jnp.pi])
        angular_gaps = wrapped_next - sorted_angles
        ideal_gap = 2.0 * jnp.pi / self.num_drones
        angular_gap_error = jnp.mean(jnp.abs(angular_gaps - ideal_gap))
        reward_angular_uniformity = -self.angular_uniformity_coef * angular_gap_error

        # Encourage swarm centroid to follow the target heading when it turns.
        target_delta = target - prev_target
        centroid_delta = centroid - prev_centroid
        target_speed = jnp.linalg.norm(target_delta)
        centroid_speed = jnp.linalg.norm(centroid_delta)
        heading_cos = jnp.where(
            jnp.logical_and(target_speed > 1e-6, centroid_speed > 1e-6),
            jnp.dot(target_delta, centroid_delta) / (target_speed * centroid_speed + 1e-6),
            0.0,
        )
        reward_heading_alignment = self.heading_alignment_coef * heading_cos

        reward_form = centroid_progress - 0.5 * radius_error - 0.25 * radius_spread
        reward_neighbor_symmetry = -self.neighbor_symmetry_coef * neighbor_distance_spread
        reward_regularize = reward_smooth_mean + reward_action_l2_mean
        team_reward = (
            reward_form
            + reward_regularize
            + reward_neighbor_symmetry
            + reward_angular_uniformity
            + reward_heading_alignment
            + collision_penalty
        ) / self.num_drones

        return (
            team_reward,
            centroid_progress,
            radius_error,
            radius_spread,
            reward_neighbor_symmetry,
            reward_angular_uniformity,
            reward_heading_alignment,
            reward_smooth_mean,
            reward_action_l2_mean,
            collision_penalty,
        )

    @override
    @partial(jit, static_argnums=(0,))
    def _sanitize_action(self, state: State, actions: jnp.ndarray) -> jnp.ndarray:
        """Clip movement to the same cubic bounds used by the renderer/map size."""
        return jnp.clip(
            state.agents_locations + actions * 0.2,
            jnp.array([-self.size, -self.size, 0]),
            jnp.array([self.size, self.size, self.size]),
        )

    def _build_target_path(self) -> jnp.ndarray:
        """Build a linear target trajectory from init to final target location."""
        num_points = self.num_ref_points
        ts = jnp.linspace(0.0, 1.0, num_points)
        path = self._init_target_location[None, :] + ts[:, None] * (self._final_target_location - self._init_target_location)[None, :]
        return path

    @partial(jit, static_argnums=(0,))
    def _sample_random_horizontal_target_path(self, key: jnp.ndarray) -> jnp.ndarray:
        """Sample one smooth horizontal target path (quadratic Bezier) from swarm spawn centroid."""
        spawn_centroid = jnp.mean(self._init_flying_pos, axis=0)
        key_height, key_theta, key_lat = random.split(key, 3)
        target_height = random.uniform(key_height, minval=self.target_height_min, maxval=self.target_height_max)
        start = jnp.array([spawn_centroid[0], spawn_centroid[1], target_height])

        theta = random.uniform(key_theta, minval=0.0, maxval=2.0 * jnp.pi)
        direction = jnp.array([jnp.cos(theta), jnp.sin(theta), 0.0])
        lateral_dir = jnp.array([-direction[1], direction[0], 0.0])

        # Clamp segment length so the end point stays within map bounds in x/y.
        eps = 1e-6
        dir_x = direction[0]
        dir_y = direction[1]
        tx = jnp.where(
            jnp.abs(dir_x) > eps,
            jnp.where(dir_x > 0.0, (self.size - start[0]) / dir_x, (-self.size - start[0]) / dir_x),
            jnp.inf,
        )
        ty = jnp.where(
            jnp.abs(dir_y) > eps,
            jnp.where(dir_y > 0.0, (self.size - start[1]) / dir_y, (-self.size - start[1]) / dir_y),
            jnp.inf,
        )
        max_reachable = jnp.minimum(tx, ty)
        segment_length = jnp.maximum(0.0, jnp.minimum(self.target_segment_length, max_reachable))
        end = start + direction * segment_length

        max_lateral = self.target_curvature_scale * segment_length
        lateral_offset = random.uniform(key_lat, minval=-max_lateral, maxval=max_lateral)
        control = (start + end) * 0.5 + lateral_dir * lateral_offset

        ts = jnp.linspace(0.0, 1.0, self.num_ref_points)
        omt = 1.0 - ts
        path = omt[:, None] ** 2 * start[None, :] + 2.0 * omt[:, None] * ts[:, None] * control[None, :] + ts[:, None] ** 2 * end[None, :]
        path = path.at[:, 2].set(target_height)
        return path

    @override
    def observation_space(self, agent: int) -> Space:
        return Box(
            low=-self.size,
            high=self.size,
            shape=(3 * (self.num_drones + 1),),  # coordinates of the drones and the target
        )

    @override
    def action_space(self, agent: int) -> Space:
        return Box(low=-1, high=1, shape=(3,))  # 3D speed vector

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_obs(self, state: State) -> jnp.ndarray:
        return jnp.append(
            # each row contains the location of one agent and the location of the target
            jnp.column_stack((state.agents_locations, jnp.tile(state.target_location, (self.num_drones, 1)))),
            # then we add agents_locations to each row without the agent which is already in the row
            # and make it only one dimension
            jnp.array([jnp.delete(state.agents_locations, agent, axis=0).flatten() for agent in range(self.num_drones)]),
            axis=1,
        )

    @override
    @partial(jit, static_argnums=(0,))
    def _transition_state(self, state: State, actions: jnp.ndarray, key: jnp.ndarray) -> State:
        path_len = state.target_path.shape[0]
        scaled_step = jnp.clip(state.timestep * self.target_speed_multiplier, 0.0, float(path_len - 1))
        lower_idx = jnp.floor(scaled_step).astype(jnp.int32)
        upper_idx = jnp.minimum(lower_idx + 1, path_len - 1)
        mix = scaled_step - lower_idx
        prev_agent_locations = state.agents_locations
        next_agent_locations = self._sanitize_action(state, actions)
        # Use realized (clipped) motion to infer the effective action applied by the env.
        effective_actions = (next_agent_locations - state.agents_locations) / 0.2
        prev_target_locations = state.target_location
        return jdc.replace(
            state,
            agents_locations=next_agent_locations,
            target_location=jnp.array([(1.0 - mix) * state.target_path[lower_idx] + mix * state.target_path[upper_idx]]),
            prev_agent_locations=prev_agent_locations,
            prev_target_locations=prev_target_locations,
            prev_actions=state.current_actions,
            current_actions=effective_actions,
        )

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_reward(self, state: State, terminations: jnp.ndarray, truncations: jnp.ndarray) -> jnp.ndarray:
        reward_crash = jnp.any(terminations) * -10 * jnp.ones(self.num_drones)
        (
            team_reward,
            centroid_progress,
            radius_error,
            radius_spread,
            reward_neighbor_symmetry,
            _reward_angular_uniformity,
            _reward_heading_alignment,
            _reward_smooth_mean,
            _reward_action_l2_mean,
            _collision_penalty,
        ) = self._formation_components(state)

        if self.multi_obj:
            formation_quality = -radius_error - 0.5 * radius_spread + reward_neighbor_symmetry
            centroid_quality = centroid_progress
            return (1 - jnp.any(terminations)) * jnp.column_stack(
                (jnp.ones(self.num_drones) * centroid_quality, jnp.ones(self.num_drones) * formation_quality)
            ) + jnp.any(terminations) * jnp.column_stack((reward_crash, reward_crash))
        else:
            return (1 - jnp.any(terminations)) * team_reward * jnp.ones(self.num_drones) + jnp.any(terminations) * reward_crash

    @override
    @partial(jit, static_argnums=(0,))
    def step(
        self, state: State, actions: jnp.ndarray, key: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict, State]:
        state = jdc.replace(state, timestep=state.timestep + 1)
        state = self._transition_state(state, actions, key)

        truncateds = self._compute_truncation(state)
        terminateds = self._compute_terminated(state)
        rewards = self._compute_reward(state, truncations=truncateds, terminations=terminateds)
        obs = self._compute_obs(state)

        (
            team_reward,
            centroid_progress,
            radius_error,
            radius_spread,
            reward_neighbor_symmetry,
            reward_angular_uniformity,
            reward_heading_alignment,
            reward_smooth_mean,
            reward_action_l2_mean,
            collision_penalty,
        ) = self._formation_components(state)

        info = {
            "reward_team": team_reward,
            "reward_centroid_progress": centroid_progress,
            "reward_radius_error": radius_error,
            "reward_radius_spread": radius_spread,
            "reward_neighbor_symmetry": reward_neighbor_symmetry,
            "reward_angular_uniformity": reward_angular_uniformity,
            "reward_heading_alignment": reward_heading_alignment,
            "reward_smooth": reward_smooth_mean,
            "reward_action_l2": reward_action_l2_mean,
            "reward_collision_penalty": collision_penalty,
        }

        return obs, rewards, terminateds, truncateds, info, state

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_terminated(self, state: State) -> jnp.ndarray:
        # collision with the ground and the target
        terminated = jnp.logical_or(
            state.agents_locations[:, 2] < CLOSENESS_THRESHOLD,
            jnp.linalg.norm(state.agents_locations - state.target_location) < CLOSENESS_THRESHOLD,
        )

        for agent in range(self.num_drones):
            distances = jnp.linalg.norm(state.agents_locations[agent] - state.agents_locations, axis=1)

            # collision between two drones
            terminated = terminated.at[agent].set(
                jnp.logical_or(terminated[agent], jnp.any(jnp.logical_and(distances > 0.001, distances < CLOSENESS_THRESHOLD)))
            )

        return jnp.any(terminated) * jnp.ones(self.num_drones)

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_truncation(self, state: State) -> jnp.ndarray:
        return (state.timestep == state.target_path.shape[0]) * jnp.ones(self.num_drones)

    @override
    @partial(jit, static_argnums=(0,))
    def reset(self, key: jnp.ndarray) -> Tuple[jnp.ndarray, dict, State]:
        zero_actions = jnp.zeros((self.num_drones, 3), dtype=self._init_flying_pos.dtype)
        target_path = jax.lax.cond(
            self.random_horizontal_path,
            lambda k: self._sample_random_horizontal_target_path(k),
            lambda k: self.target_path,
            key,
        )
        target0 = target_path[0]
        state = State(
            agents_locations=self._init_flying_pos,
            prev_agent_locations=self._init_flying_pos,
            timestep=0,
            target_location=jnp.array([target0]),
            prev_target_locations=jnp.array([target0]),
            prev_actions=zero_actions,
            current_actions=zero_actions,
            target_path=target_path,
        )
        obs = self._compute_obs(state)
        return obs, {}, state

    @override
    @partial(jit, static_argnums=(0,))
    def state(self, state: State) -> jnp.ndarray:
        return jnp.append(state.agents_locations.flatten(), state.target_location)


if __name__ == "__main__":
    from jax.lib import xla_bridge

    jax.config.update("jax_platform_name", "gpu")

    print(xla_bridge.get_backend().platform)

    num_agents = 5
    env = Escort(
        num_drones=num_agents,
        init_flying_pos=jnp.array([[0.0, 0.0, 1.0], [2.0, 1.0, 1.0], [0.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 0.0, 1.0]]),
        init_target_location=jnp.array([1.0, 1.0, 2.5]),
        final_target_location=jnp.array([-2.0, -2.0, 3.0]),
        num_intermediate_points=150,
    )

    num_envs = 1000  # number of states in parallel
    seed = 5  # test value
    key = random.PRNGKey(seed)
    key, *subkeys = random.split(key, num_envs + 1)

    # Wrappers
    env = AutoReset(env)  # Auto reset the env when done, stores additional info in the dict
    env = VecEnv(env)  # vmaps the env public methods

    obs, info, state = env.reset(jnp.stack(subkeys))

    for i in range(201):
        key, *subkeys = random.split(key, num_agents + 1)
        actions = (
            jnp.array([env.action_space(agent_id).sample(jnp.stack(subkeys[agent_id])) for agent_id in range(env.num_drones)])
            .flatten()
            .repeat(num_envs)
            .reshape((num_envs, env.num_drones, -1))
        )
        global_state = env.state(state)
        key, *subkeys = random.split(key, num_envs + 1)
        obs, rewards, term, trunc, info, state = env.step(state, actions, jnp.stack(subkeys))

        # print("obs", obs)
        print("rewards", rewards)
        # print("term", term)
        print("trunc", trunc)
        # print("info", info)
