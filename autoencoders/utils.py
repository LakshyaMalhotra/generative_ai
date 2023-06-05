import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


def plot_images(samples: torch.Tensor):
    fig, ax = plt.subplots(
        nrows=4, ncols=8, sharex=True, sharey=True, dpi=100, figsize=(8, 4)
    )
    ax = ax.flatten()
    images = samples[0].numpy().squeeze()
    for i in range(samples[0].shape[0]):
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


def train(
    dataloader: torch.utils.data.DataLoader,
    epochs: int,
    model: nn.Module,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    samples: list,
    print_every: int = 5,
) -> list:
    model = model.to(device)
    epoch_losses = []
    best_loss = np.Inf
    sample_outputs = []
    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for images, _ in dataloader:
            images = images.to(device)
            optimizer.zero_grad()
            _, out = model(images)
            loss = criterion(out, images)
            loss.backward()
            optimizer.step()
            batch_losses.append(np.round(loss.item(), 4))
        epoch_loss = np.round(np.mean(batch_losses), 4)
        epoch_losses.append(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "../models/autoencoder.pth")
        if epoch % print_every == 0:
            model.eval()
            sample_images = samples[0]
            sample_images = sample_images.to(device)
            _, outputs = model(sample_images)
            sample_outputs.append(outputs.detach().cpu().numpy())
            print(f"Epoch: [{epoch}/{epochs}], Loss: {epoch_loss}")
    return sample_outputs
