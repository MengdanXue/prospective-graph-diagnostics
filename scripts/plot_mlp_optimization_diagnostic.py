"""Plot the complete, audited validation-only MLP diagnostic."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = [
    ("raw", "Raw"),
    ("normalize_features", "NormalizeFeatures (N)"),
    ("normalize_scaled", "N × scalar"),
    ("normalize_centered", "N − train mean"),
    ("normalize_centered_scaled", "(N − train mean) × scalar"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary["record_count"] != 200 or not summary["source_content_verified"]:
        raise ValueError("plot requires the complete source-verified 200-record summary")
    outputs = [args.output_root / f"mlp_optimization_validation.{suffix}" for suffix in ("png", "svg")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("figure output already exists")
    args.output_root.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), sharey=True)
    positions = np.arange(len(CONDITIONS))
    for ax, dataset in zip(axes, ("Roman-empire", "Amazon-ratings")):
        groups = summary["aggregates"][dataset]
        for offset, decay, color, label in (
            (-0.18, "wd_5e-4", "#2476a3", "Weight decay = 0.0005"),
            (0.18, "wd_0", "#d48232", "Weight decay = 0"),
        ):
            entries = [groups[condition][decay]["validation"]["accuracy"] for condition, _ in CONDITIONS]
            means = np.array([entry["mean"] for entry in entries]) * 100
            sd = np.array([entry["sample_sd"] for entry in entries]) * 100
            ax.barh(positions + offset, means, height=0.31, xerr=sd,
                    color=color, label=label, zorder=3,
                    error_kw={"ecolor": "#30383e", "capsize": 2, "elinewidth": 0.8})
            for y, mean, error in zip(positions + offset, means, sd):
                ax.text(mean + error + 1.1, y, f"{mean:.1f}", va="center", fontsize=9)
        baseline = 100 * groups["raw"]["wd_5e-4"]["validation"]["training_majority_accuracy"]["mean"]
        ax.axvline(baseline, color="#737b80", linestyle=(0, (4, 3)), linewidth=1, zorder=2)
        ax.set_title(dataset, fontweight="bold", pad=12)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Validation accuracy (%)", labelpad=8)
        ax.set_yticks(positions, [label for _, label in CONDITIONS])
        ax.tick_params(axis="y", length=0, pad=9)
        ax.grid(axis="x", color="#e4e8eb", linewidth=0.7, zorder=0)
        ax.text(0, -0.32, f"Dashed line: training-majority baseline ({baseline:.2f}%)",
                transform=ax.transAxes, fontsize=8.5, color="#59646d")
    axes[0].invert_yaxis()
    fig.suptitle("MLP sensitivity to input scale and weight decay", x=0.04, ha="left", y=0.985,
                 fontsize=15, fontweight="bold")
    fig.text(0.04, 0.926, "Validation only · 10 paired seeds · Error bars show sample SD across seeds",
             fontsize=10, color="#59646d")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.59, 0.01),
               ncol=2, frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.225, right=0.98, bottom=0.27, top=0.80, wspace=0.17)
    fig.savefig(outputs[0], dpi=220, facecolor="white")
    fig.savefig(outputs[1], facecolor="white", metadata={"Date": None,
                "Description": f"Validation-only MLP diagnostic; config SHA256 {summary['config_sha256']}"})
    plt.close(fig)
    print(json.dumps({"figures": [str(path) for path in outputs]}))


if __name__ == "__main__":
    main()
