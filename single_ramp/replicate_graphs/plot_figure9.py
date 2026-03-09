import os
import pickle
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from single_ramp.config import BASELINE_HISTORY_PATH, REPLACEMENT_HISTORY_PATH

def plot_network_tts():
    # Load data
    with open(BASELINE_HISTORY_PATH, "rb") as f:
        baseline_data = pickle.load(f)

    with open(REPLACEMENT_HISTORY_PATH, "rb") as f:
        replacement_data = pickle.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot baseline (orange)
    ax.plot(
        baseline_data["steps"],
        baseline_data["tts"],
        color="darkorange",
        linewidth=1.2,
        alpha=0.8,
        label="Without lower bound constraint"
    )

    # Plot replacement (blue)
    ax.plot(
        replacement_data["steps"],
        replacement_data["tts"],
        color="steelblue",
        linewidth=1.2,
        alpha=0.8,
        label="With lower bound constraint"
    )

    # Formatting
    ax.set_xlim(-2000, 105000)
    ax.set_xticks([0, 50000, 100000])
    ax.set_xticklabels(["0", "50 k", "100 k"])

    # Format y-axis to scientific notation like the paper (e.g., 1e6)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.yaxis.get_offset_text().set_fontsize(12)

    ax.set_xlabel("Simulation step", fontsize=12)
    ax.set_ylabel("Total travel spend (s)", fontsize=12)

    ax.grid(True, linestyle="--", alpha=0.5)

    # Legend at the bottom
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
        ncol=1,
        fontsize=11
    )

    plt.tight_layout()
    fig_path = os.path.join("plots", "fig_9.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    plot_network_tts()