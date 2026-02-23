import matplotlib.pyplot as plt
import numpy as np

def init_plot():
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], color='tab:blue', marker='o', linestyle='-')
    ax.set_title("Live Training Performance")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Score")

    return line, ax, fig

def update_live_plot(scores, line, ax, fig):
    x = range(len(scores))
    line.set_data(x, scores)

    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()