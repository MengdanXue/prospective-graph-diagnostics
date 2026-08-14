#!/usr/bin/env python3
"""Summarize immutable discriminability records and render an audit figure."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#000000"]
MARKERS = ["o", "s", "^", "D", "x"]


def bootstrap_median_ci(values: list[float], *, samples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    medians = np.median(array[indices], axis=1)
    lower, upper = np.percentile(medians, [2.5, 97.5])
    return [float(lower), float(upper)]


def load_records(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.rglob("seed_*.json"))
    ]


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    config = record["config"]
    predictions = record.get("predictions") or {}
    errors = record.get("errors") or {}
    moments = record.get("empirical_moments") or {}
    return {
        "config_id": config["config_id"],
        "seed": record["seed"],
        "d": config["d"],
        "rho": config["rho"],
        "s": config["s"],
        "alpha": config["alpha"],
        "dimension": config["dimension"],
        "status": record["status"],
        "empirical_kappa": moments.get("empirical_kappa"),
        "exact_kappa_alpha": predictions.get("exact_kappa_alpha"),
        "exact_absolute_error": errors.get("exact_absolute_error"),
        "exact_relative_error": errors.get("exact_relative_error"),
        "rho_squared_d": predictions.get("rho_squared_d"),
        "approximation_signed_error": errors.get("approximation_signed_error"),
        "approximation_absolute_error": errors.get("approximation_absolute_error"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summary_payload(
    records: list[dict[str, Any]], specification: dict[str, Any]
) -> dict[str, Any]:
    rows = [flatten_record(record) for record in records]
    success = [row for row in rows if row["status"] == "success"]
    primary = [
        float(row["exact_relative_error"])
        for row in success
        if row["exact_relative_error"] is not None
    ]
    approximation = [
        float(row["approximation_absolute_error"])
        for row in success
        if row["approximation_absolute_error"] is not None
    ]
    summary_spec = specification["summary"]
    bootstrap_samples = summary_spec["bootstrap_samples"]
    bootstrap_seed = summary_spec["bootstrap_seed"]
    return {
        "schema_version": "1.0",
        "run_id": specification["run_id"],
        "record_counts": {
            "total": len(records),
            "success": len(success),
            "error": len(records) - len(success),
            "primary_relative_error_defined": len(primary),
            "approximation_defined": len(approximation),
        },
        "primary_estimand": {
            "name": specification["estimands"]["primary"],
            "median": float(np.median(primary)) if primary else None,
            "bootstrap_95_ci": (
                bootstrap_median_ci(
                    primary, samples=bootstrap_samples, seed=bootstrap_seed
                )
                if primary
                else None
            ),
        },
        "secondary_estimand": {
            "name": specification["estimands"]["secondary"],
            "median_absolute_error": float(np.median(approximation)) if approximation else None,
            "bootstrap_95_ci": (
                bootstrap_median_ci(
                    approximation,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + 1,
                )
                if approximation
                else None
            ),
        },
        "uncertainty": {
            "method": summary_spec["interval"],
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "unit_of_replication": "configuration-seed record",
        },
        "zero_policy": specification["estimands"]["primary_zero_policy"],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": mpl.__version__,
        },
    }


def render_figure(rows: list[dict[str, Any]], specification: dict[str, Any], output: Path) -> None:
    success = [row for row in rows if row["status"] == "success"]
    with mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(178 / 25.4, 70 / 25.4),
            layout="constrained",
        )
        ax = axes[0]
        alphas = sorted({float(row["alpha"]) for row in success})
        for index, alpha in enumerate(alphas):
            subset = [row for row in success if float(row["alpha"]) == alpha]
            marker = MARKERS[index % len(MARKERS)]
            color = COLORS[index % len(COLORS)]
            marker_colors = (
                {"color": color}
                if marker == "x"
                else {"facecolors": "none", "edgecolors": color}
            )
            ax.scatter(
                [row["exact_kappa_alpha"] for row in subset],
                [row["empirical_kappa"] for row in subset],
                s=15,
                marker=marker,
                linewidths=0.8,
                label=rf"$\alpha={alpha:g}$",
                **marker_colors,
            )
        all_values = [
            float(value)
            for row in success
            for value in (row["exact_kappa_alpha"], row["empirical_kappa"])
        ]
        upper = max(all_values) * 1.03 if all_values else 1.0
        ax.plot([0, upper], [0, upper], color="#444444", linestyle="--", linewidth=1, label="Equality")
        ax.set(
            xlim=(0, upper),
            ylim=(0, upper),
            xlabel=r"Exact $\kappa_\alpha$",
            ylabel=r"Empirical $\widehat{\kappa}_\alpha$",
            title="(a) Exact versus empirical",
        )
        ax.grid(True, color="#DDDDDD", linewidth=0.5)
        ax.legend(frameon=False, ncol=2)

        ax = axes[1]
        approximation_rows = [
            row for row in success if row["approximation_signed_error"] is not None
        ]
        grouped: dict[float, list[float]] = defaultdict(list)
        for row in approximation_rows:
            grouped[float(row["s"])].append(float(row["approximation_signed_error"]))
        bootstrap_samples = specification["summary"]["bootstrap_samples"]
        bootstrap_seed = specification["summary"]["bootstrap_seed"]
        strengths = sorted(grouped)
        medians = [float(np.median(grouped[value])) for value in strengths]
        intervals = [
            bootstrap_median_ci(
                grouped[value], samples=bootstrap_samples, seed=bootstrap_seed + index
            )
            for index, value in enumerate(strengths)
        ]
        lower = [median - interval[0] for median, interval in zip(medians, intervals)]
        upper_error = [interval[1] - median for median, interval in zip(medians, intervals)]
        for value in strengths:
            ax.scatter(
                [value] * len(grouped[value]),
                grouped[value],
                color="#777777",
                marker=".",
                s=8,
                alpha=0.35,
            )
        ax.errorbar(
            strengths,
            medians,
            yerr=[lower, upper_error],
            color="#0072B2",
            marker="o",
            linestyle="-",
            linewidth=1.2,
            capsize=2,
            label="Median and 95% bootstrap CI",
        )
        ax.axhline(0, color="#444444", linestyle="--", linewidth=1)
        ax.set(
            xlabel=r"Feature strength $s$",
            ylabel=r"Approximation bias $\rho^2d-\kappa$",
            title=r"(b) Neighbor-only approximation ($\alpha=0$)",
        )
        ax.grid(True, color="#DDDDDD", linewidth=0.5)
        ax.legend(frameon=False)

        fig.savefig(output / "validation_figure.pdf", facecolor="white", transparent=False)
        fig.savefig(
            output / "validation_figure.png",
            dpi=300,
            facecolor="white",
            transparent=False,
        )
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        print("Refusing to overwrite a non-empty summary directory.", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specification = json.loads(args.config.read_text(encoding="utf-8"))
    records = load_records(args.records_root)
    if not records:
        print("No per-seed records found.", file=sys.stderr)
        return 1
    rows = [flatten_record(record) for record in records]
    write_csv(args.output_dir / "records.csv", rows)
    summary = summary_payload(records, specification)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    render_figure(rows, specification, args.output_dir)
    manifest = {
        "schema_version": "1.0",
        "source_data": "records.csv generated from immutable per-seed JSON records",
        "transformations": [
            "No record exclusions other than status != success for plotted estimates.",
            "Relative errors with exact value at zero are omitted under the frozen zero policy.",
            "Panel b includes rho squared d only for alpha equal to zero.",
        ],
        "uncertainty": (
            f"95% percentile bootstrap CI of the median; "
            f"{specification['summary']['bootstrap_samples']} resamples; "
            f"seed {specification['summary']['bootstrap_seed']}; "
            "configuration-seed record is the replication unit."
        ),
        "missing_data": "Exact-zero relative errors are omitted and counted.",
        "figure_size_mm": [178, 70],
        "formats": ["pdf", "png"],
        "png_dpi": 300,
        "background": "opaque white",
        "palette": "Okabe-Ito-on-white subset with marker redundancy",
        "alt_text": (
            "Two-panel validation figure. Panel a compares exact and empirical "
            "discriminability ratios against an equality line for each center-feature "
            "weight. Panel b shows the signed overstatement of rho squared d relative "
            "to the exact neighbor-only ratio as feature strength increases."
        ),
    }
    (args.output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary["record_counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
