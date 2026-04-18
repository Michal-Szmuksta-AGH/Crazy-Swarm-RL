from __future__ import annotations

import argparse

from learning.execution.escort_jax.training import run_training


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO-like policy on CrazyRL Escort (JAX, vectorized).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-drones", type=int, default=4)
    parser.add_argument("--size", type=int, default=2)
    parser.add_argument("--target-height-min", type=float, default=1.0)
    parser.add_argument("--target-height-max", type=float, default=1.0)
    parser.add_argument("--target-segment-length", type=float, default=5.0)
    parser.add_argument("--target-curvature-scale", type=float, default=0.35)
    parser.add_argument("--target-speed-multiplier", type=float, default=1.0)
    parser.add_argument("--smoothness-coef", type=float, default=0.0)
    parser.add_argument("--action-l2-coef", type=float, default=0.0)
    parser.add_argument("--neighbor-symmetry-coef", type=float, default=0.0)
    parser.add_argument("--angular-uniformity-coef", type=float, default=0.0)
    parser.add_argument("--heading-alignment-coef", type=float, default=0.0)

    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=2)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--best-window", type=int, default=20)
    parser.add_argument("--save-path", type=str, default="trained_model/actor_escort_jax_parallel")
    parser.add_argument("--best-path", type=str, default="trained_model/actor_escort_jax_parallel_best")
    parser.add_argument("--metrics-dir", type=str, default="results/escort_jax_train")

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="crazyrl-escort-jax")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
