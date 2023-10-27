"""Implementation of a LeNet convolutional neural network for 1 channel input images."""

import torch.nn as nn
from torch import Tensor

class Net(nn.Module):
    """LeNet convolutional neural network for 1 channel input."""
    def __init__(self, num_classes) -> None:
        super(Net, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.full_layers = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
        self._num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(-1, 16 * 5 * 5)
        out = self.full_layers(out)
        return out
