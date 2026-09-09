import torch
import torch.nn as nn


# Custom artifact-aware interaction module (non-QKV attention)
class ArtifactInteractionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        # Extract local relationships between adjacent channel features
        self.local_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.local_bn = nn.BatchNorm1d(1)

        # Model global interactions between channel features
        self.non_local_fc = nn.Linear(channels, channels)

        self.relu = nn.ReLU(inplace=True)

        # A fixed residual scaling factor that is not trainable
        self.register_buffer("scale", torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C]
        x_reshape = x.unsqueeze(1)  # [B, 1, C]

        local_feat = self.local_conv(x_reshape)
        local_feat = self.local_bn(local_feat)
        local_feat = self.relu(local_feat)
        local_feat = local_feat.squeeze(1)  # [B, C]

        non_local_feat = self.non_local_fc(x)  # [B, C]

        out = x + self.scale * (local_feat + non_local_feat)
        return out


# Modified SE block with an ArtifactInteractionBlock
class SEBlock(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()

        hidden_channel = max(1, channel // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(channel, hidden_channel)
        self.relu = nn.ReLU(inplace=True)

        self.artifact = ArtifactInteractionBlock(hidden_channel)

        self.fc2 = nn.Linear(hidden_channel, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.shape

        # Global average pooling
        y = self.avg_pool(x).view(batch_size, channels)  # [B, C]

        y = self.fc1(y)
        y = self.relu(y)
        y = self.artifact(y)
        y = self.fc2(y)

        # Generate channel-wise attention weights
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)

        return x * y


# Adjust the channel count so that it is divisible by the specified divisor
def make_divisible(
    channels: float,
    divisor: int = 8,
    min_channels: int | None = None,
) -> int:
    if min_channels is None:
        min_channels = divisor

    new_channels = max(
        min_channels,
        int(channels + divisor / 2) // divisor * divisor,
    )

    # Prevent the rounded channel count from decreasing by more than 10%
    if new_channels < 0.9 * channels:
        new_channels += divisor

    return int(new_channels)


# Standard convolution followed by BatchNorm and LeakyReLU
def conv_bn(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    leaky: float = 0.1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


# Depthwise separable convolution
def conv_dw(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    leaky: float = 0.1,
) -> nn.Sequential:
    return nn.Sequential(
        # Depthwise convolution
        nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        ),
        nn.BatchNorm2d(in_channels),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),

        # Pointwise convolution
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


# Lightweight residual-style backbone with artifact-aware SE blocks
class ResidualNN(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        round_nearest: int = 8,
    ):
        super().__init__()

        conv1_out = make_divisible(32 * alpha, round_nearest)

        self.conv1 = conv_bn(
            in_channels=3,
            out_channels=conv1_out,
            stride=2,
        )
        self.se1 = SEBlock(conv1_out)

        channel_64 = make_divisible(64 * alpha, round_nearest)
        channel_128 = make_divisible(128 * alpha, round_nearest)
        channel_256 = make_divisible(256 * alpha, round_nearest)

        stage_channels = [
            (conv1_out, channel_64),
            (channel_64, channel_128),
            (channel_128, channel_128),
            (channel_128, channel_256),
            (channel_256, channel_256),
        ]

        stage_layers = []

        for index, (in_channels, out_channels) in enumerate(stage_channels):
            # Downsample in the second and fourth blocks
            stride = 2 if index in (1, 3) else 1

            stage_layers.append(
                conv_dw(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                )
            )

        self.stage1 = nn.Sequential(*stage_layers)

        self.se2 = SEBlock(channel_256)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(channel_256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.se1(x)

        x = self.stage1(x)
        x = self.se2(x)

        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return self.sigmoid(x)


if __name__ == "__main__":
    model = ResidualNN(alpha=1.0)

    # Example input: batch size 4, RGB image, resolution 224 × 224
    inputs = torch.randn(4, 3, 224, 224)
    outputs = model(inputs)

    print(model)
    print("Input shape:", inputs.shape)
    print("Output shape:", outputs.shape)
