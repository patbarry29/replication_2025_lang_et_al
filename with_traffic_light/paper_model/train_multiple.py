import subprocess
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

# Define the seeds you want to test
SEEDS = np.arange(20)

def run_training(use_replacement):
    variant = "replacement" if use_replacement else "baseline"
    for seed in SEEDS:
        print(f"\n{'='*40}")
        print(f"Starting {variant} run with seed {seed}")
        print(f"{'='*40}\n")

        cmd = ["python", "train.py", "--seed", str(seed)]
        if use_replacement:
            cmd.append("--use_replacement")

        # Run the process and wait for it to finish
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def load_and_plot(use_replacement, color):
    variant = "replacement" if use_replacement else "baseline"
    model_dir = "models_replacement" if use_replacement else "models"
    all_scores = []

    for seed in SEEDS:
        file_path = os.path.join(model_dir, f"training_history_{variant}_seed{seed}.pkl")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                all_scores.append(data["scores"])
        else:
            print(f"Warning: {file_path} not found.")

    if not all_scores:
        return

    # Calculate mean and standard deviation across all seeds
    scores_array = np.array(all_scores)
    mean_scores = np.mean(scores_array, axis=0)
    std_scores = np.std(scores_array, axis=0)
    episodes = np.arange(1, len(mean_scores) + 1)

    # Plot line and variance
    plt.plot(episodes, mean_scores, label=f"{variant.capitalize()} Mean", color=color)
    plt.fill_between(episodes, mean_scores - std_scores, mean_scores + std_scores, color=color, alpha=0.2)

if __name__ == "__main__":
    # 1. Execute runs sequentially
    # Comment out a line if you only want to run one variant
    run_training(use_replacement=False)
    run_training(use_replacement=True)

    # 2. Aggregate and plot results
    plt.figure(figsize=(10, 6))

    load_and_plot(use_replacement=False, color="blue")
    load_and_plot(use_replacement=True, color="orange")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("RL Training Performance Across Multiple Seeds")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("multi_seed_results.png", dpi=300)
    print("Saved plot to multi_seed_results.png")
    plt.show()