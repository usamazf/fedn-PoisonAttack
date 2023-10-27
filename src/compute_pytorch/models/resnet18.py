"""Implementation of a ResNet-18 neural network for 3."""

from torchvision.models.resnet import ResNet, BasicBlock

class Net(ResNet):
    """ResNet-18 network using default PyTorch implementation."""
    def __init__(self, num_classes) -> None:
        super(Net, self).__init__(block=BasicBlock, layers=[2,2,2,2], num_classes=num_classes)
        self._num_classes = num_classes
