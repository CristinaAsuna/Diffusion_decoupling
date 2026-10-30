from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_checkpoint_file(path: str | Path, device: torch.device) -> Any:
    path = Path(path)
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not expose the weights_only argument.
        return torch.load(path, map_location=device)


def _is_raw_state_dict(state: Any) -> bool:
    if not isinstance(state, dict) or not state:
        return False
    return all(isinstance(key, str) for key in state.keys())


def require_training_checkpoint(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError(
            "Expected a full training checkpoint with at least a 'model' entry. "
            "Use a checkpoint produced by train_research.py such as last.pt, step_xxxxxx.pt, or best_ema.pt."
        )
    return state


def select_model_state(state: Any, weights: str = "ema") -> tuple[dict[str, Any], str]:
    if isinstance(state, dict) and "model" in state:
        if weights == "ema" and "ema_model" in state:
            return state["ema_model"], "ema_model"
        return state["model"], "model"
    if _is_raw_state_dict(state):
        return state, "state_dict"
    raise ValueError("Checkpoint does not contain loadable model weights.")


def capture_rng_state() -> dict:
    import random
    import numpy as np
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng_state(state: dict | None) -> None:
    if state is None:
        return
    import random
    import numpy as np
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
