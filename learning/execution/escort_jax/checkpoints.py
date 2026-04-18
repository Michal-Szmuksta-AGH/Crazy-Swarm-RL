from __future__ import annotations

from pathlib import Path
from typing import Tuple

from flax import serialization
from flax.training.train_state import TrainState


def save_checkpoint(path: Path, actor_state: TrainState, critic_state: TrainState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"actor_params": actor_state.params, "critic_params": critic_state.params}
    path.write_bytes(serialization.to_bytes(payload))


def load_checkpoint(path: Path | str, actor_state: TrainState, critic_state: TrainState) -> Tuple[TrainState, TrainState]:
    path = Path(path)
    payload = serialization.from_bytes(
        {"actor_params": actor_state.params, "critic_params": critic_state.params},
        path.read_bytes(),
    )
    return actor_state.replace(params=payload["actor_params"]), critic_state.replace(params=payload["critic_params"])


def infer_escort_architecture_from_checkpoint(path: Path | str) -> Tuple[int, int, int]:
    path = Path(path)
    payload = serialization.msgpack_restore(path.read_bytes())
    actor_params = payload["actor_params"]
    dense_keys = sorted(k for k in actor_params["params"].keys() if k.startswith("Dense_"))
    if not dense_keys:
        raise ValueError("Could not infer architecture: no Dense_* layers found in actor checkpoint params")

    dense0_kernel = actor_params["params"][dense_keys[0]]["kernel"]
    input_dim = int(dense0_kernel.shape[0])
    hidden_dim = int(dense0_kernel.shape[1])
    # Actor has N hidden Dense layers + 1 output Dense layer.
    num_hidden_layers = len(dense_keys) - 1
    if num_hidden_layers <= 0:
        raise ValueError(f"Invalid inferred num_hidden_layers={num_hidden_layers} from checkpoint")

    # Escort observation per agent after AddIDToObs: 4 * num_drones + 3
    if (input_dim - 3) % 4 != 0:
        raise ValueError(f"Could not infer num_drones from checkpoint input_dim={input_dim}")
    num_drones = (input_dim - 3) // 4
    if num_drones <= 0:
        raise ValueError(f"Invalid inferred num_drones={num_drones} from input_dim={input_dim}")
    return num_drones, hidden_dim, num_hidden_layers
