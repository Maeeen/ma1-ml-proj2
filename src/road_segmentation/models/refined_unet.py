import torch
from torch import nn
from .unet import UNet

# We apply a base Unet model multiple times in a row, conditioned on a previous prediction.
class RefinedUnet(nn.Module):
    def __init__(self, n=3) -> None:
        super().__init__()
        
        # Main model
        self.unet = UNet(n_channels=4, n_classes=1)
        self.n = n
        
    def forward(self, x):
        # Create initial predictions
        predictions = torch.zeros_like(x[..., 0])
        
        for _ in self.n:
            concatenated = torch.concat([x, predictions])
            predictions = self.unet(concatenated)

        return predictions