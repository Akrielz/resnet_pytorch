import torch

from resnet_pytorch import ResNet, ResidualBlock


def main():
    model = ResNet(
        in_channels=3,
        num_classes=10,
        num_layers=[3, 4, 6, 3],
        num_channels=[64, 128, 256, 512],
        block=ResidualBlock
    )

    img = torch.randn(1, 3, 300, 300)
    preds = model(img)  # (1, 10)


if __name__ == "__main__":
    main()
