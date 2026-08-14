#!/usr/bin/env python3
"""Direct simulation audit of the self-neighbor discriminability formula."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np


ZERO_TOLERANCE = 1e-12


def exact_kappa_alpha(d: int, rho: float, s: float, alpha: float) -> float:
    numerator = (alpha + (1.0 - alpha) * rho) ** 2
    denominator = (
        alpha**2
        + (1.0 - alpha) ** 2 / d
        + ((1.0 - alpha) ** 2 / d) * (1.0 - rho**2) * s
    )
    return numerator / denominator


def validate_parameters(
    d: int, rho: float, s: float, alpha: float, dimension: int, samples_per_class: int
) -> None:
    if not isinstance(d, int) or isinstance(d, bool) or d < 1:
        raise ValueError("d must be an integer greater than or equal to 1")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [-1, 1]")
    if s <= 0.0:
        raise ValueError("s must be strictly positive")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    if not isinstance(samples_per_class, int) or samples_per_class <= dimension + 1:
        raise ValueError("samples_per_class must exceed dimension + 1")


def config_id(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"cfg_{hashlib.sha256(encoded).hexdigest()[:16]}"


def derived_rng_seed(seed: int, identifier: str) -> int:
    digest = hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def environment_record() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
    }


def empirical_discriminability(plus: np.ndarray, minus: np.ndarray) -> tuple[float, dict[str, Any]]:
    mean_plus = plus.mean(axis=0)
    mean_minus = minus.mean(axis=0)
    cov_plus = np.atleast_2d(np.cov(plus, rowvar=False, ddof=1))
    cov_minus = np.atleast_2d(np.cov(minus, rowvar=False, ddof=1))
    pooled_covariance = 0.5 * (cov_plus + cov_minus)
    delta = mean_plus - mean_minus
    value = float(delta @ np.linalg.solve(pooled_covariance, delta))
    moments = {
        "mean_plus": mean_plus.tolist(),
        "mean_minus": mean_minus.tolist(),
        "covariance_plus": cov_plus.tolist(),
        "covariance_minus": cov_minus.tolist(),
        "pooled_covariance": pooled_covariance.tolist(),
        "mean_difference": delta.tolist(),
        "discriminability": value,
    }
    return value, moments


def simulate_class(
    rng: np.random.Generator,
    *,
    label: int,
    d: int,
    rho: float,
    s: float,
    alpha: float,
    dimension: int,
    samples_per_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    mu = np.full(dimension, math.sqrt(s / dimension), dtype=float)
    center = label * mu + rng.standard_normal((samples_per_class, dimension))
    same_probability = (1.0 + rho) / 2.0
    same_count = rng.binomial(d, same_probability, size=samples_per_class)
    neighbor_label_mean = label * (2.0 * same_count - d) / d
    neighbor_average = (
        neighbor_label_mean[:, None] * mu
        + rng.standard_normal((samples_per_class, dimension)) / math.sqrt(d)
    )
    mixed = alpha * center + (1.0 - alpha) * neighbor_average
    return center, mixed


def base_record(
    *,
    run_id: str,
    config: dict[str, Any],
    seed: int,
    rng_seed: int,
    git_commit: str,
    samples_per_class: int,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "config": config,
        "seed": seed,
        "derived_rng_seed": rng_seed,
        "git_commit": git_commit,
        "environment": environment_record(),
        "sample_count": {
            "per_class": samples_per_class,
            "total": 2 * samples_per_class,
        },
        "empirical_moments": None,
        "predictions": None,
        "errors": None,
        "status": "pending",
        "exception": None,
    }


def run_record(
    *,
    run_id: str,
    config: dict[str, Any],
    seed: int,
    git_commit: str,
    samples_per_class: int,
) -> dict[str, Any]:
    identifier = config["config_id"]
    rng_seed = derived_rng_seed(seed, identifier)
    record = base_record(
        run_id=run_id,
        config=config,
        seed=seed,
        rng_seed=rng_seed,
        git_commit=git_commit,
        samples_per_class=samples_per_class,
    )
    try:
        d = config["d"]
        rho = config["rho"]
        s = config["s"]
        alpha = config["alpha"]
        dimension = config["dimension"]
        validate_parameters(d, rho, s, alpha, dimension, samples_per_class)

        rng = np.random.default_rng(rng_seed)
        x_plus, z_plus = simulate_class(
            rng,
            label=1,
            d=d,
            rho=rho,
            s=s,
            alpha=alpha,
            dimension=dimension,
            samples_per_class=samples_per_class,
        )
        x_minus, z_minus = simulate_class(
            rng,
            label=-1,
            d=d,
            rho=rho,
            s=s,
            alpha=alpha,
            dimension=dimension,
            samples_per_class=samples_per_class,
        )
        self_d, self_moments = empirical_discriminability(x_plus, x_minus)
        mixed_d, mixed_moments = empirical_discriminability(z_plus, z_minus)
        empirical_kappa = mixed_d / self_d
        exact = exact_kappa_alpha(d, rho, s, alpha)
        approximation = rho**2 * d if abs(alpha) <= ZERO_TOLERANCE else None

        record["empirical_moments"] = {
            "self": self_moments,
            "mixed": mixed_moments,
            "empirical_kappa": empirical_kappa,
        }
        record["predictions"] = {
            "exact_kappa_alpha": exact,
            "rho_squared_d": approximation,
        }
        record["errors"] = {
            "exact_absolute_error": abs(empirical_kappa - exact),
            "exact_relative_error": (
                abs(empirical_kappa - exact) / abs(exact)
                if abs(exact) > ZERO_TOLERANCE
                else None
            ),
            "approximation_signed_error": (
                approximation - exact if approximation is not None else None
            ),
            "approximation_absolute_error": (
                abs(approximation - exact) if approximation is not None else None
            ),
        }
        record["status"] = "success"
    except Exception as exc:  # Preserve failures as auditable records.
        record["status"] = "error"
        record["exception"] = {"type": type(exc).__name__, "message": str(exc)}
    return record


def load_grid(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expand_grid(specification: dict[str, Any]) -> list[dict[str, Any]]:
    grid = specification["grid"]
    dimension = specification["model"]["dimension"]
    configs: list[dict[str, Any]] = []
    for d, rho, s, alpha in itertools.product(
        grid["d"], grid["rho"], grid["s"], grid["alpha"]
    ):
        config = {
            "d": d,
            "rho": rho,
            "s": s,
            "alpha": alpha,
            "dimension": dimension,
        }
        config["config_id"] = config_id(config)
        configs.append(config)
    return configs


def resolve_git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def write_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--git-commit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specification = load_grid(args.config)
    configs = expand_grid(specification)
    run_id = specification["run_id"]
    samples_per_class = specification["model"]["samples_per_class"]
    git_commit = args.git_commit or resolve_git_commit(args.config.resolve().parents[1])
    targets = [
        (
            config,
            seed,
            args.output_root / run_id / config["config_id"] / f"seed_{seed}.json",
        )
        for config in configs
        for seed in specification["seeds"]
    ]
    existing = [path for _, _, path in targets if path.exists()]
    if existing:
        print(f"Refusing to overwrite {len(existing)} existing record(s).", file=sys.stderr)
        return 2

    failures = 0
    for config, seed, path in targets:
        record = run_record(
            run_id=run_id,
            config=config,
            seed=seed,
            git_commit=git_commit,
            samples_per_class=samples_per_class,
        )
        write_record(path, record)
        failures += record["status"] != "success"

    print(
        json.dumps(
            {"run_id": run_id, "records": len(targets), "failures": failures},
            sort_keys=True,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
