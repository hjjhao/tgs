import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(BasicBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.relu(x)

        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            BasicBlock(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = BasicBlock(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffX = x2.size()[3] - x1.size()[3]
        diffY = x2.size()[2] - x2.size()[2]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )
    
    def forward(self, x):
        return self.conv(x)

class Merge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Merge, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv = BasicBlock(in_channels=in_channels+1, out_channels=out_channels) # +1是拼接的depth信息
    
    def forward(self, x: torch.Tensor, depth: torch.Tensor):
        x = self.maxpool(x)
        depth_repeat = depth.repeat([1, 8]).unsqueeze(2).repeat([1, 1, 8]).unsqueeze(1)
        x_merge = torch.concat([x, depth_repeat], dim=1)

        return self.conv(x_merge)


class UNet_ResNet(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 10):
        super(UNet_ResNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = BasicBlock(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        # self.down4 = Down(256, 128)
        self.merge = Merge(256, 512)
        
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)

        self.outc = OutConv(32, n_classes)

    def forward(self, xx):
        x = xx[0]
        depth = xx[1]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        x5 = self.merge(x4, depth)


        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

if __name__ == '__main__':
    # x = torch.randn((1, 3, 128, 128))
    # y = torch.randn(1)
    net = UNet_ResNet().to('cuda')
    # summary(net, (3, 128, 128))
    # print('x.shape: ', x.shape) # [3, 1]
    # x = x.repeat([1, 8]).unsqueeze(2).repeat([1, 1, 8]).unsqueeze(1)
    # print('x.shape: ', x.shape) # [3, 8]
    # print(x)
    # print([x, y][1])