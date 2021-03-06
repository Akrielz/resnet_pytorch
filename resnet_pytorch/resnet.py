from typing import List, Optional, Type, Union

from einops import rearrange
from torch import nn

from resnet_pytorch.bottleneck_block import BottleneckBlock
from resnet_pytorch.conv_block import ConvBlock
from resnet_pytorch.residual_block import ResidualBlock


class ResNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 10,
            num_layers: Optional[List[int]] = None,
            num_channels: Optional[List[int]] = None,
            block: Type[Union[ResidualBlock, BottleneckBlock]] = None,
    ):
        super(ResNet, self).__init__()

        # assign default values to parameters
        if block is None:
            block = ResidualBlock

        if num_layers is None:
            num_layers = [3, 4, 6, 3]

        if num_channels is None:
            num_channels = [64, 128, 256, 512]

        # save parameters
        self.num_layers = num_layers
        self.num_channels = num_channels

        # build layers
        self.conv1 = ConvBlock(in_channels, num_channels[0], stride=2, kernel_size=7, padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = self.__build_layers(block)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.LazyLinear(num_classes)

    def __build_layers(
            self,
            block: Type[Union[ResidualBlock, BottleneckBlock]]
    ):
        layers = nn.ModuleList()
        for i in range(len(self.num_layers)):
            prev_num_channels = self.num_channels[i - 1] if i != 0 else self.num_channels[i]
            layers.append(self.__build_layer(i, prev_num_channels, block))

        return layers

    def __build_layer(
            self,
            index: int,
            prev_num_channels: int,
            block: Type[Union[ResidualBlock, BottleneckBlock]],
    ):
        layer = nn.ModuleList()
        num_layers = self.num_layers[index]
        num_channels = self.num_channels[index]

        for j in range(num_layers):
            stride = 2 if j == 0 and prev_num_channels != num_channels else 1
            input_num_channels = prev_num_channels if j == 0 else num_channels

            layer.append(block(input_num_channels, num_channels, stride=stride))

        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.max_pool(out)

        for layer in self.layers:
            out = layer(out)

        out = self.avg_pool(out)
        out = rearrange(out, "b ... -> b (...)")
        out = self.fc(out)

        return out


def build_resnet18(num_classes: int = 10, in_channels: int = 3):
    return ResNet(
        in_channels=in_channels,
        num_classes=num_classes,
        num_layers=[2, 2, 2, 2],
        num_channels=[64, 128, 256, 512],
        block=ResidualBlock
    )


def build_resnet34(num_classes: int = 10, in_channels: int = 3):
    return ResNet(
        in_channels=in_channels,
        num_classes=num_classes,
        num_layers=[3, 4, 6, 3],
        num_channels=[64, 128, 256, 512],
        block=ResidualBlock
    )


def build_resnet50(num_classes: int = 10, in_channels: int = 3):
    return ResNet(
        in_channels=in_channels,
        num_classes=num_classes,
        num_layers=[3, 4, 6, 3],
        num_channels=[64, 128, 256, 512],
        block=BottleneckBlock
    )


def build_resnet101(num_classes: int = 10, in_channels: int = 3):
    return ResNet(
        in_channels=in_channels,
        num_classes=num_classes,
        num_layers=[3, 4, 23, 3],
        num_channels=[64, 128, 256, 512],
        block=BottleneckBlock
    )


def build_resnet152(num_classes: int = 10, in_channels: int = 3):
    return ResNet(
        in_channels=in_channels,
        num_classes=num_classes,
        num_layers=[3, 8, 36, 3],
        num_channels=[64, 256, 512, 1024],
        block=BottleneckBlock
    )
