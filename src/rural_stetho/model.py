import torch
from torch import nn

class DualViewRespiratoryNet(nn.Module):
    """CNN over log-Mel audio fused with acoustic summary features."""
    def __init__(self,n_classes,n_descriptors=36):
        super().__init__()
        self.spec=nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.GELU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.GELU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.GELU(),
            nn.AdaptiveAvgPool2d((4,8)),nn.Flatten(),nn.Linear(2048,128),nn.GELU(),nn.Dropout(.25))
        self.acoustic=nn.Sequential(nn.Linear(n_descriptors,64),nn.BatchNorm1d(64),nn.GELU(),nn.Dropout(.15),nn.Linear(64,32),nn.GELU())
        self.head=nn.Sequential(nn.Linear(160,64),nn.GELU(),nn.Dropout(.3),nn.Linear(64,n_classes))
    def forward(self,mel,desc):
        return self.head(torch.cat([self.spec(mel.unsqueeze(1)),self.acoustic(desc)],dim=1))
