import os
import pickle
import matplotlib.pyplot as plt

def plot_replacement_percentage():
    replacement_path = os.path.join(r"C:\Users","pbarry","Documents","2025_yang_dqn","with_traffic_light","paper_model","models", "training_history_replacement.pkl")

    # Load data
    with open(replacement_path, "rb") as f:
        data = pickle.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the replacement percentage (red line like the paper)
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

    # Adjust X-axis limit based on how many steps you actually simulated
    max_step = max(data["steps"])
    ax.set_xlim(-5000, max_step + 5000)

    ax.set_xlabel("Simulation step", fontsize=12)
    ax.set_ylabel("Action replacement percentage (%)", fontsize=12)

    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(r"C:\Users\pbarry\Documents\2025_yang_dqn\with_traffic_light\paper_model\replicate_graphs\plots\fig_10.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    plot_replacement_percentage()