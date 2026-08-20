"""Render the frozen prospective regret--coverage sensitivity figure.

The script reads only the published diagnostic audit. It performs no model
training, rescoring, filtering, smoothing, or statistical inference.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "results"
    / "diagnostic"
    / "route_a_prospective_v2"
    / "analysis"
    / "diagnostic_audit.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "diagnostic"
    / "route_a_prospective_v2"
    / "analysis"
    / "prospective_regret_coverage.pdf"
)

POLICIES = {
    "historical_combined": {
        "label": "Combined",
        "color": "#000000",
        "linestyle": "-",
        "marker": "o",
        "zorder": 4,
    },
    "always_graph": {
        "label": "Always-graph (full set)",
        "color": "#0072B2",
        "linestyle": "--",
        "marker": "s",
        "zorder": 3,
    },
    "always_mlp": {
        "label": "Always-MLP (full set)",
        "color": "#D55E00",
        "linestyle": "-.",
        "marker": "^",
        "zorder": 2,
    },
    "validation_selection": {
        "label": "Validation selection",
        "color": "#009E73",
        "linestyle": ":",
        "marker": "D",
        "zorder": 3,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output and its PNG companion.",
    )
    return parser.parse_args()


def _validated_curve(payload: dict, method: str) -> tuple[list[float], list[float]]:
    try:
        curve = payload["methods"][method]["risk_coverage_curve"]
    except KeyError as exc:
        raise ValueError(f"missing frozen curve for {method}") from exc
    if not curve:
        raise ValueError(f"empty frozen curve for {method}")

    coverage = [100.0 * float(point["coverage"]) for point in curve]
    regret = [100.0 * float(point["covered_mean_regret"]) for point in curve]
    if any(right <= left for left, right in zip(coverage, coverage[1:])):
        raise ValueError(f"coverage is not strictly increasing for {method}")
    if any(value < 0.0 for value in regret):
        raise ValueError(f"negative regret in frozen curve for {method}")
    return coverage, regret


def _display_curve(payload: dict, method: str) -> tuple[list[float], list[float]]:
    """Return only the identifiable coverage display for a frozen policy."""
    coverage, regret = _validated_curve(payload, method)
    curve = payload["methods"][method]["risk_coverage_curve"]
    confidence = [float(point["minimum_confidence"]) for point in curve]
    if len(set(confidence)) == 1:
        return [coverage[-1]], [regret[-1]]
    return coverage, regret


def render(input_path: Path, output_path: Path, *, force: bool) -> None:
    input_path = input_path.resolve(strict=True)
    output_path = output_path.resolve()
    png_path = output_path.with_suffix(".png")
    for path in (output_path, png_path):
        if path.exists() and not force:
            raise FileExistsError(f"refusing to overwrite {path}; pass --force")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if payload.get("status") != "scored":
        raise ValueError("diagnostic audit is not in the scored state")
    counts = payload.get("record_counts", {})
    if counts.get("evaluation_units") != 110:
        raise ValueError("expected the frozen 110-unit audit")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(
        figsize=(140.0 / 25.4, 84.0 / 25.4),
        layout="constrained",
        facecolor="white",
    )
    for method, style in POLICIES.items():
        coverage, regret = _display_curve(payload, method)
        markevery = max(1, len(coverage) // 9)
        ax.plot(
            coverage,
            regret,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"] if len(coverage) > 1 else "none",
            linewidth=1.6,
            marker=style["marker"],
            markersize=3.8,
            markerfacecolor="white",
            markeredgewidth=0.9,
            markevery=markevery,
            zorder=style["zorder"],
        )
        if len(coverage) > 1:
            ax.plot(
                coverage[-1],
                regret[-1],
                color=style["color"],
                marker=style["marker"],
                markersize=5.0,
                linestyle="none",
                zorder=style["zorder"] + 1,
            )

    combined_x, combined_y = _validated_curve(payload, "historical_combined")
    ax.annotate(
        "Combined: 68.2% coverage",
        xy=(combined_x[-1], combined_y[-1]),
        xytext=(54.0, 4.1),
        arrowprops={"arrowstyle": "->", "color": "#444444", "linewidth": 0.8},
        color="#222222",
        fontsize=7.5,
    )

    ax.set(
        xlim=(0.0, 100.0),
        ylim=(0.0, 12.0),
        xlabel="Coverage after confidence ordering (%)",
        ylabel="Covered-set mean raw-accuracy regret (pp)",
    )
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=2, handlelength=2.8)

    fixed_time = datetime(2026, 8, 20, tzinfo=UTC)
    pdf_tmp = output_path.with_suffix(".pdf.tmp")
    png_tmp = png_path.with_suffix(".png.tmp")
    try:
        fig.savefig(
            pdf_tmp,
            format="pdf",
            metadata={
                "Title": "Prospective regret--coverage sensitivity",
                "Author": "Mengdan Xue",
                "Creator": "scripts/plot_prospective_regret_coverage.py",
                "CreationDate": fixed_time,
                "ModDate": fixed_time,
            },
        )
        fig.savefig(png_tmp, format="png", dpi=600, metadata={"Software": "Matplotlib"})
        os.replace(pdf_tmp, output_path)
        os.replace(png_tmp, png_path)
    finally:
        plt.close(fig)
        for path in (pdf_tmp, png_tmp):
            if path.exists():
                path.unlink()


def main() -> None:
    args = parse_args()
    render(args.input, args.output, force=args.force)


if __name__ == "__main__":
    main()
