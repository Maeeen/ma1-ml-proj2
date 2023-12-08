import torch
from torch import nn
from .unet import UNet

class RefinedUnet(nn.Module):
    def __init__(self, n=3, net=None) -> None:
        super().__init__()
        
        # Main model
        self.unet = net if net is not None else UNet(n_channels=4, n_classes=1)
        self.n = n
        
    def forward(self, x, stages=False):
        # Create initial predictions
        predictions = torch.zeros_like(x[:, 0, ...]).unsqueeze(1)
        history = [predictions]
        
        for _ in range(self.n):
            concatenated = torch.cat([x, predictions], dim=1).clone()
            predictions = self.unet(concatenated).unsqueeze(1)
            history.append(predictions)

        return history if stages else predictions