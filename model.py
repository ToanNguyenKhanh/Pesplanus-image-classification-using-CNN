"""
@author: <nktoan163@gmail.com>
"""
import torch.nn as nn
from dataset import PesplanusDataset
class CNN(nn.Module):
    def __init__(self, num_classes=len(PesplanusDataset().classes)):
        super(CNN, self).__init__()
        self.conv1 = self.block(in_channels=3, out_channels=16)
        self.conv2 = self.block(in_channels=16, out_channels=32)
        self.conv3 = self.block(in_channels=32, out_channels=64)
        self.conv4 = self.block(in_channels=64, out_channels=128)
        self.conv5 = self.block(in_channels=128, out_channels=128)
        self.conv6 = self.block(in_channels=128, out_channels=128)

        self.fullyConnected1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=6272, out_features=num_classes),
            nn.LeakyReLU()
        )

    def block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.shape[0], -1)
        x = self.fullyConnected1(x)
        return x

