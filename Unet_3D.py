import torch
import torch.nn as nn


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        """两个不改变尺寸的卷积"""

        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        """ 下采样两倍并 卷积两次"""
        super(Down, self).__init__(
            nn.MaxPool3d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True):
        """将第一个输入上采样两倍后和第二个输入拼接并卷积两次"""
        super(Up, self).__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv3d(in_channels, num_classes, kernel_size=1)
        )


class Unet_encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 base_c,
                 trilinear: bool = True,
                 ):
        super(Unet_encoder, self).__init__()
        self.in_channels = in_channels
        self.trilinear = trilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 16)
        factor = 2 if trilinear else 1
        self.down5 = Down(base_c * 16, base_c * 32 // factor)

    def forward(self, x):
        x1 = self.in_conv(x)  # [base_c, 64， 64， 48]
        x2 = self.down1(x1)  # [base_c * 2, 32， 32， 24]
        x3 = self.down2(x2)  # [base_c * 4, 16， 16， 12]
        x4 = self.down3(x3)  # [base_c * 8, 8， 8， 6]
        x5 = self.down4(x4)  # [base_c * 16, 4， 4， 3]
        final_feature_map = self.down5(x5)

        return final_feature_map, x5, x4, x3, x2, x1


class Unet_decoder(nn.Module):
    def __init__(self,
                 num_classes: int,
                 base_c: int,
                 trilinear: bool = True,
                 ):
        super(Unet_decoder, self).__init__()
        self.num_classes = num_classes
        self.trilinear = trilinear

        factor = 2 if trilinear else 1

        self.up5 = Up(base_c * 32, base_c * 16 // factor, trilinear)
        self.up4 = Up(base_c * 16, base_c * 8 // factor, trilinear)
        self.up3 = Up(base_c * 8, base_c * 4 // factor, trilinear)
        self.up2 = Up(base_c * 4, base_c * 2 // factor, trilinear)
        self.up1 = Up(base_c * 2, base_c, trilinear)
        self.Aortic_out_conv = OutConv(base_c, num_classes)

    def forward(self, final_feature_map, x5, x4, x3, x2, x1):

        x = self.up5(final_feature_map, x5)  # weight of size [256, 512, 3, 3, 3], expected input[2, 1024, 8, 8, 16] to have 512 channels, but got 1024 channels instead
        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        logits = self.Aortic_out_conv(x)

        return logits

class UNet(nn.Module):
    def __init__(self, in_channel, num_class, base_c):
        super(UNet, self).__init__()
        self.encoder = Unet_encoder(in_channels=in_channel, base_c=base_c)
        self.decoder = Unet_decoder(num_classes=num_class, base_c=base_c)

    def forward(self, input):
        final_feature_map, x5, x4, x3, x2, x1 = self.encoder(input)
        pre = self.decoder(final_feature_map, x5, x4, x3, x2, x1)

        return pre


if __name__ == "__main__":
    x = torch.ones((2, 1, 128, 128, 256))
    model = UNet(in_channel=1, num_class=1, base_c=8)
    y = model(x)
    print(y.shape)