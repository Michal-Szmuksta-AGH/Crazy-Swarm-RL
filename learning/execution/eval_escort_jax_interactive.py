from __future__ import annotations

import argparse
from pathlib import Path

import distrax
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import optax
from flax.training.train_state import TrainState

from crazy_rl.multi_agent.jax.base_parallel_env import CLOSENESS_THRESHOLD
from crazy_rl.multi_agent.jax.escort import Escort
from learning.execution.escort_jax.checkpoints import infer_escort_architecture_from_checkpoint, load_checkpoint
from learning.execution.escort_jax.env_setup import make_init_positions
from learning.execution.escort_jax.models import Actor, Critic
from learning.execution.escort_jax.renderer import Pygame3DRenderer


def _normalize_obs(obs: jnp.ndarray, size: float) -> jnp.ndarray:
    # Matches NormalizeObservation wrapper behavior for Escort's symmetric Box space.
    return obs / size


def _add_agent_ids(obs: jnp.ndarray, num_drones: int) -> jnp.ndarray:
    ids = jnp.eye(num_drones, dtype=obs.dtype)
    return jnp.concatenate([obs, ids], axis=1)


def _target_velocity_from_keys(renderer, speed_mps: float) -> np.ndarray:
    keys = renderer._pygame.key.get_pressed()

    # Arrow keys: horizontal control, PageUp/PageDown: vertical control.
    vx = 0.0
    vy = 0.0
    vz = 0.0
    if keys[renderer._pygame.K_UP]:
        vx += 1.0
    if keys[renderer._pygame.K_DOWN]:
        vx -= 1.0
    if keys[renderer._pygame.K_RIGHT]:
        vy += 1.0
    if keys[renderer._pygame.K_LEFT]:
        vy -= 1.0
    if keys[renderer._pygame.K_PAGEUP]:
        vz += 1.0
    if keys[renderer._pygame.K_PAGEDOWN]:
        vz -= 1.0

    v = np.array([vx, vy, vz], dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm > 1e-6:
        v = v / norm * float(speed_mps)
    return v


def run_interactive_evaluation(args) -> None:
    num_drones, hidden_dim, num_hidden_layers = infer_escort_architecture_from_checkpoint(args.model_path)

    init_flying_pos = make_init_positions(num_drones)
    init_target = jnp.array([args.target_start_x, args.target_start_y, args.target_start_z], dtype=jnp.float32)

    # final_target_location/target_path are not used in interactive mode, but Escort requires them.
    env = Escort(
        num_drones=num_drones,
        init_flying_pos=init_flying_pos,
        init_target_location=init_target,
        final_target_location=init_target,
        target_speed_multiplier=0.0,
        smoothness_coef=args.smoothness_coef,
        action_l2_coef=args.action_l2_coef,
        neighbor_symmetry_coef=args.neighbor_symmetry_coef,
        size=args.size,
    )

    obs_dim = env.observation_space(0).shape[0] + num_drones
    action_dim = env.action_space(0).shape[0]
    global_dim = env.state(env.reset(jax.random.PRNGKey(0))[2]).shape[-1]

    actor = Actor(action_dim=action_dim, hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers)
    critic = Critic(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers)

    key = jax.random.PRNGKey(args.seed)
    key, actor_key, critic_key = jax.random.split(key, 3)

    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, jnp.zeros((obs_dim,), dtype=jnp.float32)),
        tx=optax.adam(1e-3),
    )
    critic_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic.init(critic_key, jnp.zeros((global_dim,), dtype=jnp.float32)),
        tx=optax.adam(1e-3),
    )

    actor_state, critic_state = load_checkpoint(Path(args.model_path), actor_state, critic_state)

    renderer = Pygame3DRenderer(
        window_size=args.window_size,
        map_size=args.size,
        fps=args.render_fps,
        camera_distance=args.camera_distance,
        camera_yaw_deg=args.camera_yaw,
        camera_pitch_deg=args.camera_pitch,
    )

    print("Interactive controls: Arrow keys move target in horizontal plane, PageUp/PageDown move target in Z.")
    print("Press Q or ESC to quit.")

    key, reset_key = jax.random.split(key)
    _, _, state = env.reset(reset_key)
    # Keep target at requested start regardless of default reset path.
    state = jdc.replace(
        state,
        target_location=jnp.array([init_target]),
        prev_target_locations=jnp.array([init_target]),
    )

    step = 0
    try:
        while not renderer.closed and step < args.max_steps:
            step += 1

            target_pos = np.asarray(state.target_location[0], dtype=np.float32)
            dt = 1.0 / max(1, renderer.fps)
            target_vel = _target_velocity_from_keys(renderer, args.target_control_speed)
            new_target = target_pos + target_vel * dt
            new_target = np.clip(new_target, [-args.size, -args.size, 0.0], [args.size, args.size, args.size])

            state = jdc.replace(
                state,
                target_location=jnp.array([new_target], dtype=jnp.float32),
                prev_target_locations=state.target_location,
            )

            raw_obs = env._compute_obs(state)
            norm_obs = _normalize_obs(raw_obs, args.size)
            obs_with_ids = _add_agent_ids(norm_obs, num_drones)

            mean, log_std = jax.vmap(lambda x: actor.apply(actor_state.params, x))(obs_with_ids)
            if args.deterministic:
                actions = mean
            else:
                clipped_log_std = jnp.clip(log_std, args.stochastic_log_std_min, args.stochastic_log_std_max)
                scaled_std = jnp.exp(clipped_log_std) * args.stochastic_std_scale
                dist = distrax.MultivariateNormalDiag(mean, scaled_std)
                key, sample_key = jax.random.split(key)
                actions = dist.sample(seed=sample_key)

            next_agents = env._sanitize_action(state, actions)
            effective_actions = (next_agents - state.agents_locations) / 0.2
            state = jdc.replace(
                state,
                timestep=state.timestep + 1,
                prev_agent_locations=state.agents_locations,
                agents_locations=next_agents,
                prev_actions=state.current_actions,
                current_actions=effective_actions,
            )

            # Keep the session alive, resetting drones only for ground/drone-drone collisions.
            ground_collision = jnp.any(state.agents_locations[:, 2] < CLOSENESS_THRESHOLD)
            drone_collision = jnp.any(
                jnp.array(
                    [
                        jnp.any(
                            jnp.logical_and(
                                jnp.linalg.norm(state.agents_locations[i] - state.agents_locations, axis=1) > 0.001,
                                jnp.linalg.norm(state.agents_locations[i] - state.agents_locations, axis=1)
                                < CLOSENESS_THRESHOLD,
                            )
                        )
                        for i in range(num_drones)
                    ]
                )
            )
            if bool(jnp.logical_or(ground_collision, drone_collision)):
                state = jdc.replace(
                    state,
                    agents_locations=env._init_flying_pos,
                    prev_agent_locations=env._init_flying_pos,
                    prev_actions=jnp.zeros_like(state.prev_actions),
                    current_actions=jnp.zeros_like(state.current_actions),
                )

            agent_xyz = np.asarray(state.agents_locations, dtype=np.float32)
            target_xyz_single = np.asarray(state.target_location, dtype=np.float32)
            target_xyz = np.repeat(target_xyz_single, num_drones, axis=0)

            if not renderer.render(
                agent_xyz,
                target_xyz,
                title=(
                    f"Escort Interactive | step {step} | "
                    f"target=({target_xyz_single[0, 0]:.2f}, {target_xyz_single[0, 1]:.2f}, {target_xyz_single[0, 2]:.2f})"
                ),
            ):
                break
    finally:
        renderer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive Escort evaluation: control target with keyboard.")
    parser.add_argument("--model-path", type=str, default="trained_model/actor_escort_jax_parallel_best")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--target-start-x", type=float, default=0.0)
    parser.add_argument("--target-start-y", type=float, default=0.0)
    parser.add_argument("--target-start-z", type=float, default=1.0)
    parser.add_argument("--target-control-speed", type=float, default=1.0, help="Target speed in m/s under key press")
    parser.add_argument("--max-steps", type=int, default=50_000)

    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--stochastic-std-scale", type=float, default=0.25)
    parser.add_argument("--stochastic-log-std-min", type=float, default=-2.5)
    parser.add_argument("--stochastic-log-std-max", type=float, default=-0.2)

    parser.add_argument("--smoothness-coef", type=float, default=0.0)
    parser.add_argument("--action-l2-coef", type=float, default=0.0)
    parser.add_argument("--neighbor-symmetry-coef", type=float, default=0.0)

    parser.add_argument("--render-fps", type=int, default=20)
    parser.add_argument("--window-size", type=int, default=1800)
    parser.add_argument("--camera-distance", type=float, default=5.5)
    parser.add_argument("--camera-yaw", type=float, default=-70.0)
    parser.add_argument("--camera-pitch", type=float, default=30.0)
    return parser.parse_args()


if __name__ == "__main__":
    run_interactive_evaluation(parse_args())
