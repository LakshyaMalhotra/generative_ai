import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_images(indices: np.ndarray, samples: torch.Tensor):
    fig, ax = plt.subplots(
        nrows=4, ncols=8, sharex=True, sharey=True, dpi=100, figsize=(8, 8)
    )
    ax = ax.flatten()
    images = samples[0].numpy().squeeze()
    for i in range(len(indices)):
        image, label = (images[i], samples[1][i].item())
        ax[i].imshow(image, cmap="gray")
        ax[i].set_title(f"label:{label}")
        ax[i].set_axis_off()

    fig.tight_layout()
    plt.show()


def plot_outputs(outputs: list, labels: torch.Tensor, indices: np.ndarray):
    fig, ax = plt.subplots(
        nrows=10, ncols=10, sharex=True, sharey=True, dpi=100, figsize=(10, 12)
    )
    outputs = np.array(outputs).squeeze()
    for i in range(10):
        for j in range(10):
            idx = indices[j]
            ax[i][j].imshow(outputs[i, idx, :, :], cmap="gray")
            if i == 0:
                ax[i][j].set_title(f"label:{labels[idx].item()}")
            ax[i][j].set_axis_off()
    fig.tight_layout()
    plt.show()
