import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CIFAR10Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # C1 Block - Using Depthwise Separable Conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # First layer can't be depthwise
            nn.BatchNorm2d(16),
            nn.ReLU(),
            DepthwiseSeparableConv(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        # C2 Block - Depthwise Separable Convolution
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(32, 48, kernel_size=3),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            DepthwiseSeparableConv(48, 80, kernel_size=3),
            nn.BatchNorm2d(80),
            nn.ReLU(),
        )
        
        # C3 Block - Dilated Convolution with Depthwise
        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv(80, 96, kernel_size=3, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 112, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(112),
            nn.ReLU(),
        )
        
        # C4 Block - Strided Convolution
        self.conv4 = nn.Sequential(
            DepthwiseSeparableConv(112, 120, kernel_size=3, stride=2),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            DepthwiseSeparableConv(120, 120, kernel_size=3),
            nn.BatchNorm2d(120),
            nn.ReLU(),
        )
        
        # Global Average Pooling and Final FC Layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(120, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(-1, 120)
        x = self.fc(x)
        return x 