import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores):
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
