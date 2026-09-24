"""Plot the exact retained-record cutoff curves; no model execution."""
import argparse
import json
from pathlib import Path


def plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if output.exists():
        raise FileExistsError(output)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    order = ["full", "GAT", "GCN", "GPR-GNN", "GraphSAGE", "H2GCN", "LINKX"]
    fig, axes = plt.subplots(4, 2, figsize=(10.5, 8.5), sharex=True, layout="constrained")
    for name, ax in zip(order, axes.flat):
        row = summary["portfolios"][name]
        curve = row["exact_threshold_curve"]
        # Each non-singleton interval is (lower, upper], except at t=0.
        # Horizontal segments preserve the exact cutoff direction; vertical
        # connectors are visual guides, not values at an observed threshold.
        previous = None
        for interval in curve:
            lower, upper = interval["lower"], interval["upper"]
            value = interval["mean_regret_pp"]
            ax.plot([lower, upper], [value, value], color="#176b87", lw=1.7)
            if previous is not None:
                ax.plot([lower, lower], [previous, value], color="#176b87", lw=0.8)
            previous = value
        ax.axhline(row["headroom_pp"], color="#b05c23", ls="--", lw=1.1)
        ax.axvline(.55, color="#777777", ls=":", lw=1.0)
        minimum = row["policy_regret_pp"]["test_oracle_h1_cutoff"]
        ax.set_title(("Six architectures" if name == "full" else name)
                     + f"  |  minimum {minimum:.3f} pp", fontsize=11, loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Mean regret (pp)")
        ax.grid(axis="y", alpha=.18)
        ax.tick_params(labelbottom=True)
    for ax in axes[-1]:
        ax.set_xlabel(r"Cutoff $t$ (graph if $h_1\geq t$)")
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(handles=[
        Line2D([0], [0], color="#176b87", lw=1.7, label="Exact cutoff curve"),
        Line2D([0], [0], color="#b05c23", ls="--", label="Always-graph / headroom"),
        Line2D([0], [0], color="#777777", ls=":", label="Frozen cutoff: 0.55"),
    ], loc="upper left", frameon=False)
    axes.flat[-1].text(.04, .42,
        "Same 110 retained units per panel.\nEach portfolio has its own oracle.\n"
        "Panel y-axis scales differ.\nMinima use test outcomes: optimistic bounds.",
        transform=axes.flat[-1].transAxes, va="top", fontsize=11, linespacing=1.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"Title": "Exact post-hoc homophily cutoff curves",
                                 "Author": "", "CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plot(json.loads(args.input.read_text(encoding="utf-8")), args.output)
