"""Implementation of a simple multilayer perceptron network."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class Net(nn.Module):
    """Multilayer percenptron (MLP) network."""
    def __init__(self, num_classes) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1024, 24)
        self.fc2 = nn.Linear(24, num_classes)
        self._num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
