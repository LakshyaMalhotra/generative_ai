import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=16),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=32),
            nn.Flatten(1, -1),
            nn.Linear(32, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            View((32, 1, 1)),
            nn.ConvTranspose2d(32, 32, 3, 2),
            nn.BatchNorm2d(num_features=32),
            nn.ConvTranspose2d(32, 32, 3, 2),
            nn.BatchNorm2d(num_features=32),
            nn.ConvTranspose2d(32, 16, 2, 2),
            nn.BatchNorm2d(num_features=16),
            nn.ConvTranspose2d(16, 1, 2, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        enc_out = self.encoder(x)
        out = self.decoder(enc_out)
        return enc_out, out


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, inpt):
        batch_size = inpt.size(0)
        shape = (batch_size, *self.shape)
        return inpt.view(shape)


if __name__ == "__main__":
    x = torch.rand(2, 1, 28, 28)
    model = AutoEncoder()
    y1, y2 = model(x)
    print(y1.shape, y2.shape)
