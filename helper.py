import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores):
    """Plots a live graph showing training progress based on a list of scores.

    This function updates a live plot to visualize the training process, displaying
    a graph of the given scores. It clears any previous plots, re-plots the new data,
    and updates the display in real-time.

    Args:
        scores (list of float): A list of scores to plot.

    """
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Epochs')
    plt.ylabel('MSE Loss')
    plt.plot(scores)
    plt.ylim(ymin=0)
    plt.show(block=False)
    plt.pause(.1)
