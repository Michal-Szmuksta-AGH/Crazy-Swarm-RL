from __future__ import annotations

import json
from pathlib import Path

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from learning.execution.escort_jax.checkpoints import infer_escort_architecture_from_checkpoint, load_checkpoint
from learning.execution.escort_jax.env_setup import make_escort_env
from learning.execution.escort_jax.models import Actor, Critic
from learning.execution.escort_jax.renderer import Pygame3DRenderer


def _unwrap_base_state(state):
    """Unwrap nested wrapper states to the raw Escort state."""
    s = state
    while hasattr(s, "env_state"):
        s = s.env_state
    return s


def _extract_physical_xyz(state):
    """Return raw (meters) agent and target coordinates for the first vectorized env."""
    base_state = _unwrap_base_state(state)
    agent_xyz = np.asarray(base_state.agents_locations)[0]
    target_xyz = np.asarray(base_state.target_location)[0]
    return agent_xyz, target_xyz


def run_evaluation(args) -> None:
    num_drones, hidden_dim, num_hidden_layers = infer_escort_architecture_from_checkpoint(args.model_path)
    init_target = jnp.array([args.target_init_x, args.target_init_y, args.target_init_z], dtype=jnp.float32)
    final_target = jnp.array([args.target_final_x, args.target_final_y, args.target_final_z], dtype=jnp.float32)

    env = make_escort_env(
        num_drones=num_drones,
        size=args.size,
        gamma=0.99,
        init_target_location=init_target,
        final_target_location=final_target,
        num_intermediate_points=198,
        target_speed_multiplier=args.target_speed_multiplier,
        smoothness_coef=args.smoothness_coef,
        action_l2_coef=args.action_l2_coef,
        neighbor_symmetry_coef=args.neighbor_symmetry_coef,
        normalize_reward=args.normalize_reward,
    )

    print(f"Evaluation reward mode: {'normalized' if args.normalize_reward else 'raw'}")
    print(f"JAX backend: {jax.default_backend()} | devices: {jax.devices()}")
    if not args.deterministic:
        print(
            "Stochastic eval controls: "
            f"std_scale={args.stochastic_std_scale}, "
            f"log_std_clamp=[{args.stochastic_log_std_min}, {args.stochastic_log_std_max}]"
        )

    obs_dim = env.observation_space(0).shape[0] + num_drones
    action_dim = env.action_space(0).shape[0]
    global_dim = env.state(env.reset(jnp.stack([jax.random.PRNGKey(0)]))[2]).shape[-1]

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    renderer = Pygame3DRenderer(
        window_size=args.window_size,
        map_size=args.size,
        fps=args.render_fps,
        camera_distance=args.camera_distance,
        camera_yaw_deg=args.camera_yaw,
        camera_pitch_deg=args.camera_pitch,
    )

    episode_returns = []
    episode_lengths = []

    try:
        for episode in range(1, args.episodes + 1):
            key, *subkeys = jax.random.split(key, 2)
            obs, info, state = env.reset(jnp.stack(subkeys))
            done = np.array([False])
            ep_return = 0.0
            ep_steps = 0
            done_reason = "running"
            _, target_xyz = _extract_physical_xyz(state)
            start_target = target_xyz[0].copy()
            last_target = start_target.copy()

            while not bool(done[0]):
                # Read physical coordinates before stepping; AutoReset may overwrite state on done.
                agent_xyz, target_xyz = _extract_physical_xyz(state)
                last_target = target_xyz[0].copy()
                target_xyz = np.repeat(target_xyz, num_drones, axis=0)

                obs_flat = obs.reshape((-1, obs.shape[-1]))
                mean, log_std = jax.vmap(lambda x: actor.apply(actor_state.params, x))(obs_flat)
                if args.deterministic:
                    actions_flat = mean
                else:
                    clipped_log_std = jnp.clip(log_std, args.stochastic_log_std_min, args.stochastic_log_std_max)
                    scaled_std = jnp.exp(clipped_log_std) * args.stochastic_std_scale
                    dist = distrax.MultivariateNormalDiag(mean, scaled_std)
                    key, action_key = jax.random.split(key)
                    actions_flat = dist.sample(seed=action_key)
                actions = actions_flat.reshape((1, num_drones, action_dim))

                keep_open = renderer.render(
                    agent_xyz,
                    target_xyz,
                    title=(
                        f"Escort JAX | ep {episode}/{args.episodes} | step {ep_steps + 1} "
                        f"| return {ep_return:.2f}"
                    ),
                )
                if not keep_open:
                    print("Pygame window closed by user.")
                    break

                key, step_key = jax.random.split(key)
                obs, rewards, term, trunc, info, state = env.step(state, actions, jnp.stack([step_key]))
                term_np = np.asarray(term)
                trunc_np = np.asarray(trunc)
                done = np.logical_or(np.any(term_np, axis=-1), np.any(trunc_np, axis=-1))
                if bool(done[0]):
                    done_reason = "terminated" if bool(np.any(term_np[0])) else "truncated"
                ep_return += float(np.asarray(rewards).sum())
                ep_steps += 1

            if renderer.closed:
                break

            episode_returns.append(ep_return)
            episode_lengths.append(ep_steps)
            moved_distance = float(np.linalg.norm(last_target - start_target))
            planned_distance = float(np.linalg.norm(final_target - init_target))
            print(
                f"Episode {episode}/{args.episodes} return={ep_return:.3f} steps={ep_steps} "
                f"done={done_reason} last_target=({last_target[0]:.3f}, {last_target[1]:.3f}, {last_target[2]:.3f}) "
                f"moved_distance={moved_distance:.3f}m planned_distance={planned_distance:.3f}m"
            )
    finally:
        renderer.close()

    if not episode_returns:
        print("No completed episodes to summarize.")
        return

    summary = {
        "episodes": len(episode_returns),
        "return_mean": float(np.mean(episode_returns)),
        "return_std": float(np.std(episode_returns)),
        "return_min": float(np.min(episode_returns)),
        "return_max": float(np.max(episode_returns)),
        "steps_mean": float(np.mean(episode_lengths)),
        "steps_min": int(np.min(episode_lengths)),
        "steps_max": int(np.max(episode_lengths)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved evaluation summary to", output_dir / "summary.json")
    print(json.dumps(summary, indent=2))
