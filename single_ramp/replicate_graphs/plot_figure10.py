import os
import pickle
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import REPLACEMENT_HISTORY_PATH

def plot_replacement_percentage():
    # Load data
    with open(REPLACEMENT_HISTORY_PATH, "rb") as f:
        data = pickle.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the replacement percentage
    ax.plot(
        data["steps"],
        data["replacement_pct"],
        color="firebrick",
        linewidth=1.2,
        alpha=0.9
    )

    # Formatting matching Figure 10
    ax.set_ylim(-1, 25)
    ax.set_yticks([0, 5, 10, 15, 20, 25])

    max_step = max(data["steps"])
    ax.set_xlim(-5000, max_step + 5000)

    ax.set_xlabel("Simulation step", fontsize=12)
    ax.set_ylabel("Action replacement percentage (%)", fontsize=12)

    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join("plots", "fig_10.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    plot_replacement_percentage()