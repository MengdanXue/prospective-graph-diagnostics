#!/usr/bin/env python3
"""Numerically verify the self-feature mixing discriminability formula."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class FormulaResult:
    alpha: float
    rho: float
    degree: int
    strength: float
    theory_mean_coefficient: float
    base_noise_coefficient: float
    label_mixture_coefficient: float
    denominator: float
    ratio: float


def validate_parameters(alpha: float, rho: float, degree: int, strength: float) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [-1, 1]")
    if degree < 1:
        raise ValueError("degree must be a positive integer")
    if strength <= 0.0 or not math.isfinite(strength):
        raise ValueError("strength must be finite and strictly positive")


def formula(alpha: float, rho: float, degree: int, strength: float) -> FormulaResult:
    validate_parameters(alpha, rho, degree, strength)
    neighbor_weight = 1.0 - alpha
    mean_coefficient = alpha + neighbor_weight * rho
    base_noise = alpha**2 + neighbor_weight**2 / degree
    label_mixture = neighbor_weight**2 * (1.0 - rho**2) / degree
    denominator = base_noise + label_mixture * strength
    return FormulaResult(
        alpha=alpha,
        rho=rho,
        degree=degree,
        strength=strength,
        theory_mean_coefficient=mean_coefficient,
        base_noise_coefficient=base_noise,
        label_mixture_coefficient=label_mixture,
        denominator=denominator,
        ratio=mean_coefficient**2 / denominator,
    )


def simulate(
    alpha: float,
    rho: float,
    degree: int,
    strength: float,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    if samples < 2:
        raise ValueError("samples must be at least two")
    theory = formula(alpha, rho, degree, strength)
    rng = np.random.default_rng(seed)
    mu = math.sqrt(strength)

    self_features = mu + rng.standard_normal(samples)
    same_label_probability = (1.0 + rho) / 2.0
    neighbor_labels = np.where(
        rng.random((samples, degree)) < same_label_probability,
        1.0,
        -1.0,
    )
    neighbor_features = neighbor_labels * mu + rng.standard_normal((samples, degree))
    neighbor_average = neighbor_features.mean(axis=1)
    mixed = alpha * self_features + (1.0 - alpha) * neighbor_average

    result = asdict(theory)
    result.update(
        samples=samples,
        seed=seed,
        empirical_mean_coefficient=float(mixed.mean() / mu),
        empirical_variance=float(mixed.var(ddof=0)),
        theory_variance=theory.denominator,
    )
    return result


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--degree", type=int, required=True)
    parser.add_argument("--strength", type=float, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    formula_parser = subparsers.add_parser("formula")
    add_common_arguments(formula_parser)

    simulation_parser = subparsers.add_parser("simulate")
    add_common_arguments(simulation_parser)
    simulation_parser.add_argument("--samples", type=int, required=True)
    simulation_parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "formula":
            result = asdict(formula(args.alpha, args.rho, args.degree, args.strength))
        else:
            result = simulate(
                args.alpha,
                args.rho,
                args.degree,
                args.strength,
                args.samples,
                args.seed,
            )
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        return 2

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
