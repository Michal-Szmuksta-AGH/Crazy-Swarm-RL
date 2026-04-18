from __future__ import annotations

import argparse

from learning.execution.escort_jax.evaluation import run_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Escort JAX checkpoint with realtime pygame 3D animation.")
    parser.add_argument("--model-path", type=str, default="trained_model/actor_escort_jax_parallel_best")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--size", type=int, default=2)
    parser.add_argument("--target-init-x", type=float, default=-1.0)
    parser.add_argument("--target-init-y", type=float, default=-1.0)
    parser.add_argument("--target-init-z", type=float, default=1.0)
    parser.add_argument("--target-final-x", type=float, default=1.0)
    parser.add_argument("--target-final-y", type=float, default=1.0)
    parser.add_argument("--target-final-z", type=float, default=1.0)
    parser.add_argument("--target-speed-multiplier", type=float, default=1.0)
    parser.add_argument("--smoothness-coef", type=float, default=0.0)
    parser.add_argument("--action-l2-coef", type=float, default=0.0)
    parser.add_argument("--neighbor-symmetry-coef", type=float, default=0.0)

    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--stochastic-std-scale",
        type=float,
        default=0.25,
        help="Scale factor for policy std during stochastic evaluation; lower values reduce chaos.",
    )
    parser.add_argument(
        "--stochastic-log-std-min",
        type=float,
        default=-2.5,
        help="Lower clamp for log_std during stochastic evaluation.",
    )
    parser.add_argument(
        "--stochastic-log-std-max",
        type=float,
        default=-0.2,
        help="Upper clamp for log_std during stochastic evaluation.",
    )
    parser.add_argument("--normalize-reward", action="store_true")

    parser.add_argument("--render-fps", type=int, default=20)
    parser.add_argument("--window-size", type=int, default=1400)
    parser.add_argument("--camera-distance", type=float, default=5.5)
    parser.add_argument("--camera-yaw", type=float, default=-70.0)
    parser.add_argument("--camera-pitch", type=float, default=30.0)

    parser.add_argument("--output-dir", type=str, default="results/escort_jax_eval")
    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
