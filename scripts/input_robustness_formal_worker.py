"""One child-process formal core worker used by the outer guardian."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from scripts.input_robustness_training_core import run_model_unit_with_checkpoints


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    # The request is an exclusive, parent-created torch envelope so tensors
    # and device bindings survive the process boundary without a JSON shim.
    request = torch.load(args.request, map_location="cpu", weights_only=False)
    if not isinstance(request, dict) or not isinstance(request.get("core_kwargs"), dict):
        raise ValueError("formal worker request must contain core_kwargs")
    delay = float(request.get("worker_sleep_seconds", 0.0))
    if delay < 0 or delay > 3600:
        raise ValueError("worker_sleep_seconds is outside the guarded range")
    if delay:
        time.sleep(delay)
    result = run_model_unit_with_checkpoints(**request["core_kwargs"])
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

