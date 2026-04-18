from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, NamedTuple

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from learning.execution.escort_jax.checkpoints import save_checkpoint
from learning.execution.escort_jax.env_setup import make_escort_env
from learning.execution.escort_jax.metrics import plot_training_curves, save_metrics_csv
from learning.execution.escort_jax.models import Actor, Critic


class Batch(NamedTuple):
    terminated: jnp.ndarray
    actions: jnp.ndarray
    values: jnp.ndarray
    rewards: jnp.ndarray
    raw_rewards: jnp.ndarray
    log_probs: jnp.ndarray
    obs: jnp.ndarray
    global_obs: jnp.ndarray
    reward_team: jnp.ndarray
    reward_centroid_progress: jnp.ndarray
    reward_radius_error: jnp.ndarray
    reward_radius_spread: jnp.ndarray
    reward_neighbor_symmetry: jnp.ndarray
    reward_angular_uniformity: jnp.ndarray
    reward_heading_alignment: jnp.ndarray
    reward_smooth: jnp.ndarray
    reward_action_l2: jnp.ndarray
    reward_collision_penalty: jnp.ndarray


def _rollout_action(actor: Actor, actor_params, obs: jnp.ndarray, key: jax.Array):
    obs_flat = obs.reshape((-1, obs.shape[-1]))
    mean, log_std = jax.vmap(lambda x: actor.apply(actor_params, x))(obs_flat)
    dist = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
    actions_flat = dist.sample(seed=key)
    log_probs_flat = dist.log_prob(actions_flat)
    entropy = dist.entropy().mean()
    return actions_flat.reshape(obs.shape[0], obs.shape[1], -1), log_probs_flat.reshape(obs.shape[0], obs.shape[1]), entropy


def run_training(args) -> None:
    key = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)
    print(f"JAX backend: {jax.default_backend()} | devices: {jax.devices()}")

    # Training uses random horizontal target segments per episode (generalization across headings and heights).
    target_height_mid = 0.5 * (args.target_height_min + args.target_height_max)
    init_target = jnp.array([0.0, 0.0, target_height_mid], dtype=jnp.float32)
    final_target = jnp.array([min(args.target_segment_length, args.size), 0.0, target_height_mid], dtype=jnp.float32)

    env = make_escort_env(
        num_drones=args.num_drones,
        size=args.size,
        gamma=args.gamma,
        init_target_location=init_target,
        final_target_location=final_target,
        num_intermediate_points=198,
        target_speed_multiplier=args.target_speed_multiplier,
        random_horizontal_path=True,
        target_height_min=args.target_height_min,
        target_height_max=args.target_height_max,
        target_segment_length=args.target_segment_length,
        target_curvature_scale=args.target_curvature_scale,
        smoothness_coef=args.smoothness_coef,
        action_l2_coef=args.action_l2_coef,
        neighbor_symmetry_coef=args.neighbor_symmetry_coef,
        angular_uniformity_coef=args.angular_uniformity_coef,
        heading_alignment_coef=args.heading_alignment_coef,
        normalize_reward=True,
    )
    raw_env = make_escort_env(
        num_drones=args.num_drones,
        size=args.size,
        gamma=args.gamma,
        init_target_location=init_target,
        final_target_location=final_target,
        num_intermediate_points=198,
        target_speed_multiplier=args.target_speed_multiplier,
        random_horizontal_path=True,
        target_height_min=args.target_height_min,
        target_height_max=args.target_height_max,
        target_segment_length=args.target_segment_length,
        target_curvature_scale=args.target_curvature_scale,
        smoothness_coef=args.smoothness_coef,
        action_l2_coef=args.action_l2_coef,
        neighbor_symmetry_coef=args.neighbor_symmetry_coef,
        angular_uniformity_coef=args.angular_uniformity_coef,
        heading_alignment_coef=args.heading_alignment_coef,
        normalize_reward=False,
    )

    key, *subkeys = jax.random.split(key, args.num_envs + 1)
    obs, info, state = env.reset(jnp.stack(subkeys))
    raw_obs, raw_info, raw_state = raw_env.reset(jnp.stack(subkeys))

    obs_dim = env.observation_space(0).shape[0] + args.num_drones
    action_dim = env.action_space(0).shape[0]
    global_dim = env.state(state).shape[-1]

    actor = Actor(action_dim=action_dim, hidden_dim=args.hidden_dim, num_hidden_layers=args.num_hidden_layers)
    critic = Critic(hidden_dim=args.hidden_dim, num_hidden_layers=args.num_hidden_layers)
    key, actor_key, critic_key = jax.random.split(key, 3)

    dummy_obs = jnp.zeros((obs_dim,), dtype=jnp.float32)
    dummy_global_obs = jnp.zeros((global_dim,), dtype=jnp.float32)
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, dummy_obs),
        tx=optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adam(args.actor_lr)),
    )
    critic_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic.init(critic_key, dummy_global_obs),
        tx=optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adam(args.critic_lr)),
    )

    num_updates = max(1, args.total_timesteps // (args.num_envs * args.num_steps))
    metrics_rows: List[dict] = []
    norm_history: List[float] = []
    raw_history: List[float] = []
    best_raw_rolling_return = -float("inf")
    best_update = -1
    start_time = time.time()

    wandb_run = None
    if args.use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        except Exception as exc:
            print(f"W&B disabled because initialization failed: {exc}")
            wandb_run = None

    @jax.jit
    def rollout_step(actor_params, critic_params, obs, state, raw_state, key):
        key, policy_key, step_key = jax.random.split(key, 3)
        actions, log_probs, entropy = _rollout_action(actor, actor_params, obs, policy_key)
        global_obs = env.state(state)
        values = jax.vmap(lambda x: critic.apply(critic_params, x))(global_obs)
        step_keys = jax.random.split(step_key, args.num_envs)

        next_obs, rewards, term, trunc, info, next_state = env.step(state, actions, jnp.stack(step_keys))
        next_raw_obs, raw_rewards, raw_term, raw_trunc, raw_info, next_raw_state = raw_env.step(
            raw_state,
            actions,
            jnp.stack(step_keys),
        )
        terminated = jnp.logical_or(jnp.any(term, axis=-1), jnp.any(trunc, axis=-1))

        transition = Batch(
            terminated=terminated,
            actions=actions,
            values=values,
            rewards=rewards.sum(axis=-1),
            raw_rewards=raw_rewards.sum(axis=-1),
            log_probs=log_probs,
            obs=obs,
            global_obs=global_obs,
            reward_team=raw_info["reward_team"],
            reward_centroid_progress=raw_info["reward_centroid_progress"],
            reward_radius_error=raw_info["reward_radius_error"],
            reward_radius_spread=raw_info["reward_radius_spread"],
            reward_neighbor_symmetry=raw_info["reward_neighbor_symmetry"],
            reward_angular_uniformity=raw_info["reward_angular_uniformity"],
            reward_heading_alignment=raw_info["reward_heading_alignment"],
            reward_smooth=raw_info["reward_smooth"],
            reward_action_l2=raw_info["reward_action_l2"],
            reward_collision_penalty=raw_info["reward_collision_penalty"],
        )
        return key, next_obs, next_state, next_raw_state, transition, entropy

    @jax.jit
    def compute_gae(traj_batch: Batch, last_values: jnp.ndarray):
        def scan_fn(carry, transition):
            gae, next_value = carry
            delta = transition.rewards + args.gamma * next_value * (1 - transition.terminated) - transition.values
            gae = delta + args.gamma * args.gae_lambda * (1 - transition.terminated) * gae
            return (gae, transition.values), gae

        _, advantages = jax.lax.scan(scan_fn, (jnp.zeros_like(last_values), last_values), traj_batch, reverse=True)
        returns = advantages + traj_batch.values
        return advantages, returns

    @jax.jit
    def update_step(actor_state, critic_state, obs_batch, actions_batch, old_log_probs_batch, advantages_batch, returns_batch, global_obs_batch):
        obs_flat = obs_batch.reshape((-1, obs_dim))
        actions_flat = actions_batch.reshape((-1, action_dim))
        old_log_probs_flat = old_log_probs_batch.reshape((-1,))
        advantages_flat = jnp.repeat(advantages_batch, args.num_drones, axis=-1).reshape((-1,))
        returns_flat = returns_batch.reshape((-1,))
        global_obs_flat = global_obs_batch.reshape((-1, global_dim))

        def loss_fn(actor_params, critic_params):
            mean, log_std = jax.vmap(lambda x: actor.apply(actor_params, x))(obs_flat)
            dist = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
            new_log_probs = dist.log_prob(actions_flat)
            ratio = jnp.exp(new_log_probs - old_log_probs_flat)
            unclipped = ratio * advantages_flat
            clipped = jnp.clip(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * advantages_flat
            actor_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
            values = jax.vmap(lambda x: critic.apply(critic_params, x))(global_obs_flat)
            critic_loss = jnp.mean(jnp.square(values - returns_flat))
            entropy = jnp.mean(dist.entropy())
            total_loss = actor_loss + args.value_coef * critic_loss - args.entropy_coef * entropy
            return total_loss, (actor_loss, critic_loss, entropy)

        (loss_value, (actor_loss, critic_loss, entropy)), grads = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(
            actor_state.params,
            critic_state.params,
        )
        actor_state = actor_state.apply_gradients(grads=grads[0])
        critic_state = critic_state.apply_gradients(grads=grads[1])
        return actor_state, critic_state, actor_loss, critic_loss, entropy

    try:
        for update in range(1, num_updates + 1):
            traj: List[Batch] = []
            rollout_entropy = []

            for _ in range(args.num_steps):
                key, obs, state, raw_state, transition, entropy = rollout_step(
                    actor_state.params,
                    critic_state.params,
                    obs,
                    state,
                    raw_state,
                    key,
                )
                traj.append(transition)
                rollout_entropy.append(float(entropy))

            traj_batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *traj)
            last_values = jax.vmap(lambda x: critic.apply(critic_state.params, x))(env.state(state))
            advantages, returns = compute_gae(traj_batch, last_values)

            for _ in range(args.update_epochs):
                actor_state, critic_state, actor_loss, critic_loss, entropy = update_step(
                    actor_state,
                    critic_state,
                    traj_batch.obs,
                    traj_batch.actions,
                    traj_batch.log_probs,
                    advantages,
                    returns,
                    traj_batch.global_obs,
                )

            norm_return = float(jnp.mean(traj_batch.rewards))
            raw_return = float(jnp.mean(traj_batch.raw_rewards))
            norm_history.append(norm_return)
            raw_history.append(raw_return)
            rolling_window = max(1, args.best_window)
            rolling_norm = float(np.mean(norm_history[-rolling_window:]))
            rolling_raw = float(np.mean(raw_history[-rolling_window:]))
            elapsed = time.time() - start_time
            sps = int((update * args.num_envs * args.num_steps) / max(elapsed, 1e-9))

            row = {
                "update": update,
                "return": norm_return,
                "raw_return": raw_return,
                "reward_team": float(jnp.mean(traj_batch.reward_team)),
                "reward_centroid_progress": float(jnp.mean(traj_batch.reward_centroid_progress)),
                "reward_radius_error": float(jnp.mean(traj_batch.reward_radius_error)),
                "reward_radius_spread": float(jnp.mean(traj_batch.reward_radius_spread)),
                "reward_neighbor_symmetry": float(jnp.mean(traj_batch.reward_neighbor_symmetry)),
                "reward_angular_uniformity": float(jnp.mean(traj_batch.reward_angular_uniformity)),
                "reward_heading_alignment": float(jnp.mean(traj_batch.reward_heading_alignment)),
                "reward_smooth": float(jnp.mean(traj_batch.reward_smooth)),
                "reward_action_l2": float(jnp.mean(traj_batch.reward_action_l2)),
                "reward_collision_penalty": float(jnp.mean(traj_batch.reward_collision_penalty)),
                "actor_loss": float(actor_loss),
                "critic_loss": float(critic_loss),
                "entropy": float(jnp.mean(jnp.asarray(rollout_entropy))),
                "rolling_return": rolling_norm,
                "raw_rolling_return": rolling_raw,
                "sps": sps,
                "elapsed_sec": float(elapsed),
            }
            metrics_rows.append(row)

            if wandb_run is not None:
                wandb_run.log(row)

            if rolling_raw > best_raw_rolling_return:
                best_raw_rolling_return = rolling_raw
                best_update = update
                save_checkpoint(Path(args.best_path), actor_state, critic_state)

            if update == 1 or update % args.log_every == 0:
                print(
                    f"update={update:04d} return={norm_return:8.3f} raw_return={raw_return:8.3f} "
                    f"team={row['reward_team']:7.3f} centroid_prog={row['reward_centroid_progress']:7.3f} "
                    f"rad_err={row['reward_radius_error']:7.3f} nn_sym={row['reward_neighbor_symmetry']:7.3f} "
                    f"ang_uni={row['reward_angular_uniformity']:7.3f} head={row['reward_heading_alignment']:7.3f} "
                    f"actor_loss={float(actor_loss):8.4f} critic_loss={float(critic_loss):8.4f} "
                    f"entropy={float(row['entropy']):7.4f} rolling={rolling_norm:8.3f} "
                    f"raw_rolling={rolling_raw:8.3f} sps={sps}"
                )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    save_checkpoint(Path(args.save_path), actor_state, critic_state)
    print(f"Saved model to {args.save_path}")

    metrics_dir = Path(args.metrics_dir)
    save_metrics_csv(metrics_dir / "metrics.csv", metrics_rows)
    plot_training_curves(metrics_dir / "training_curves.png", metrics_rows)
    summary = {
        "best_update": best_update,
        "best_raw_rolling_return": best_raw_rolling_return,
        "best_checkpoint": args.best_path,
        "last_checkpoint": args.save_path,
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved best checkpoint to {args.best_path}")
    print(f"Saved metrics to {metrics_dir / 'metrics.csv'}")
    print(f"Saved plot to {metrics_dir / 'training_curves.png'}")
    print(f"Saved summary to {metrics_dir / 'summary.json'}")
