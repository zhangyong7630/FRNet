"""
Author: zhangyong
email: zhangyong7630@163.com
"""

from typing import Tuple, Union
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.upsample import UpSample
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import InterpolateMode, UpsampleMode


def get_conv_layer(in_ch: int, out_ch: int, ksize: int = 3, stride: int = 1, bias: bool = False):
    return Convolution(3, in_ch, out_ch, strides=stride, kernel_size=ksize, bias=bias, conv_only=True)


def get_upsample_layer(in_ch: int, out_ch: int, upsample_mode: Union[UpsampleMode, str] = "nontrainable",
                       scale_factor: int = 2):
    return UpSample(
        spatial_dims=3,
        in_channels=in_ch,
        out_channels=out_ch,
        scale_factor=scale_factor,
        mode=upsample_mode,
        interp_mode=InterpolateMode.LINEAR,
        align_corners=False,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ksize: int = 3,
                 # norm: Union[Tuple, str] = ("GROUP", {"num_groups": 8}),
                 norm: Union[Tuple, str] = ("instance", {"affine": True}),
                 act: Union[Tuple, str] = ("RELU", {"inplace": True}),
                 ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            get_conv_layer(in_ch=in_ch, out_ch=out_ch, ksize=ksize),
            get_norm_layer(name=norm, spatial_dims=3, channels=out_ch),
            get_act_layer(act),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, in_ch: int, ksize: int = 3,
                 norm: Union[Tuple, str] = ("instance", {"affine": True}),
                 # norm: Union[Tuple, str] = ("GROUP", {"num_groups": 8}),
                 act: Union[Tuple, str] = ("RELU", {"inplace": True}),
                 ) -> None:
        super().__init__()

        if ksize % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")
        self.conv = nn.Sequential(
            get_conv_layer(in_ch=in_ch, out_ch=in_ch),
            get_norm_layer(name=norm, spatial_dims=3, channels=in_ch),
            get_act_layer(act),
            get_conv_layer(in_ch=in_ch, out_ch=in_ch),
            get_norm_layer(name=norm, spatial_dims=3, channels=in_ch),
        )

        self.act = get_act_layer(act)

    def forward(self, x):
        identity = x
        x = self.conv(x)
        x += identity
        x = self.act(x)
        return x


class PPM(nn.Module):
    def __init__(self, in_ch: int,
                 norm: Union[Tuple, str] = ("instance", {"affine": True}),
                 # norm: Union[Tuple, str] = ("GROUP", {"num_groups": 8}),
                 act: Union[Tuple, str] = ("RELU", {"inplace": True}),
                 upsample_mode: Union[UpsampleMode, str] = "nontrainable"
                 ) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            get_conv_layer(in_ch=in_ch, out_ch=in_ch, ksize=3),
            get_norm_layer(name=norm, spatial_dims=3, channels=in_ch),
            get_act_layer(act),
        )  # 1 down-sample

        self.conv2_1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=3, stride=2, padding=1),
            get_conv_layer(in_ch=in_ch, out_ch=in_ch, ksize=3),
            get_norm_layer(name=norm, spatial_dims=3, channels=in_ch),
            get_act_layer(act),
            get_upsample_layer(in_ch=in_ch, out_ch=in_ch, upsample_mode=upsample_mode, scale_factor=2)
        )  # 2 down-sample
        self.conv2_2 = nn.Sequential(
            get_conv_layer(in_ch=in_ch, out_ch=in_ch, ksize=1),
            get_norm_layer(name=norm, spatial_dims=3, channels=in_ch),
            get_act_layer(act),
        )

        self.conv3_1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=5, stride=4, padding=2),
            get_conv_layer(in_ch=in_ch, out_ch=in_ch, ksize=3),
            get_norm_layer(name=norm, spatial_dims=3, channels=in_ch),
            get_act_layer(act),
            get_upsample_layer(in_ch=in_ch, out_ch=in_ch, upsample_mode="nontrainable", scale_factor=4)
        )  # 4 down-sample
        self.conv3_2 = nn.Sequential(
            get_conv_layer(in_ch=in_ch, out_ch=in_ch, ksize=1),
            get_norm_layer(name=norm, spatial_dims=3, channels=in_ch),
            get_act_layer(act),
        )

        self.fusion = nn.Sequential(
            get_conv_layer(in_ch=in_ch, out_ch=in_ch, ksize=3),
            get_norm_layer(name=norm, spatial_dims=3, channels=in_ch),
            get_act_layer(act)

        )

    def forward(self, x):
        conv1 = self.conv1(x)
        # 2 down-sample
        conv2 = self.conv2_1(x)
        conv2 = conv2 + conv1
        conv2 = self.conv2_2(conv2)

        # 4 down-sample
        conv3 = self.conv3_1(x)
        conv3 = conv3 + conv1
        conv3 = self.conv3_2(conv3)

        # 进行信息的融合
        conv = conv1 + conv2 + conv3
        conv = self.fusion(conv)

        return conv


class DownSample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 norm: Union[Tuple, str] = ("instance", {"affine": True}),
                 act: Union[Tuple, str] = ("RELU", {"inplace": True})) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            get_conv_layer(in_ch=in_ch, out_ch=out_ch, ksize=3, stride=2),
            get_norm_layer(name=norm, spatial_dims=3, channels=out_ch),
            get_act_layer(act),
        )

    def forward(self, x):
        out = self.conv1(x)
        return out


class FRNet(nn.Module):
    # feature reserve network for Chorioid Plexus (CPs) segmentation in 7T MRI.

    def __init__(
            self,
            in_ch: int = 1,
            out_ch: int = 2,
            init_filters: int = 32,
            act: Union[Tuple, str] = ("RELU", {"inplace": True}),
            norm: Union[Tuple, str] = ("instance", {"affine": True}),
            blocks_down: tuple = (1, 2, 2, 2),
            upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            ConvBlock(in_ch=in_ch, out_ch=init_filters),
            *[ResBlock(in_ch=init_filters * 2 ** 0, norm=norm, act=act) for _ in range(blocks_down[0])]
        )  # 1/1
        self.encoder2 = nn.Sequential(
            DownSample(in_ch=init_filters * 2 ** 0, out_ch=init_filters * 2 ** 1),
            *[ResBlock(in_ch=init_filters * 2 ** 1, norm=norm, act=act) for _ in range(blocks_down[1])]
        )  # 1/2
        self.encoder3 = nn.Sequential(
            DownSample(in_ch=init_filters * 2 ** 1, out_ch=init_filters * 2 ** 2),
            *[ResBlock(in_ch=init_filters * 2 ** 2, norm=norm, act=act) for _ in range(blocks_down[2])]
        )  # 1/4
        self.encoder4 = nn.Sequential(
            DownSample(in_ch=init_filters * 2 ** 2, out_ch=init_filters * 2 ** 3),
            *[ResBlock(in_ch=init_filters * 2 ** 3, norm=norm, act=act) for _ in range(blocks_down[3])]
        )  # 1/8

        self.ppm = PPM(in_ch=init_filters * 2 ** 3)

        # Decoder
        # stage 4->3
        self.up3 = nn.Sequential(
            ConvBlock(in_ch=init_filters * 2 ** 3, out_ch=init_filters * 2 ** 2, norm=norm, act=act, ksize=1),
            get_upsample_layer(init_filters * 2 ** 2, out_ch=init_filters * 2 ** 2, upsample_mode=upsample_mode),
        )  # 1/8 ->1/4

        self.up_fusion3 = nn.Sequential(
            *[ResBlock(in_ch=init_filters * 2 ** 2, norm=norm, act=act) for _ in range(1)]
        )  # 1/4

        # stage 3->2
        self.up2 = nn.Sequential(
            ConvBlock(in_ch=init_filters * 2 ** 2, out_ch=init_filters * 2 ** 1, norm=norm, act=act, ksize=1),
            get_upsample_layer(init_filters * 2 ** 1, out_ch=init_filters * 2 ** 1, upsample_mode=upsample_mode),
        )  # 1/4 ->1/2

        self.up_fusion2 = nn.Sequential(
            *[ResBlock(in_ch=init_filters * 2 ** 1, norm=norm, act=act) for _ in range(1)]
        )  # 1/2

        # stage 2->1
        self.up1 = nn.Sequential(
            ConvBlock(in_ch=init_filters * 2 ** 1, out_ch=init_filters * 2 ** 0, norm=norm, act=act, ksize=1),
            get_upsample_layer(init_filters * 2 ** 0, out_ch=init_filters * 2 ** 0, upsample_mode=upsample_mode),
        )  # 1/2 ->1/1

        self.up_fusion1 = nn.Sequential(
            *[ResBlock(in_ch=init_filters * 2 ** 0, norm=norm, act=act) for _ in range(1)]
        )  # 1/1

        self.conv_final = get_conv_layer(in_ch=init_filters, out_ch=out_ch, ksize=3)

        # details information flows

        self.dp1 = nn.Sequential(nn.AvgPool3d(kernel_size=3, stride=2, padding=1),
                                 ConvBlock(in_ch=init_filters, out_ch=init_filters * 2 ** 1, ksize=1)
                                 )
        self.dp1_fusion = ConvBlock(in_ch=init_filters * 2 ** 1, out_ch=init_filters * 2 ** 1, ksize=1)

        self.dp2 = nn.Sequential(nn.AvgPool3d(kernel_size=5, stride=4, padding=2),
                                 ConvBlock(in_ch=init_filters, out_ch=init_filters * 2 ** 2, ksize=1)
                                 )
        self.dp2_fusion = ConvBlock(in_ch=init_filters * 2 ** 2, out_ch=init_filters * 2 ** 2, ksize=1)

        self.dp3 = nn.Sequential(nn.AvgPool3d(kernel_size=7, stride=8, padding=2),
                                 ConvBlock(in_ch=init_filters, out_ch=init_filters * 2 ** 3, ksize=1)
                                 )
        self.dp3_fusion = ConvBlock(in_ch=init_filters * 2 ** 3, out_ch=init_filters * 2 ** 3, ksize=3)

        # need process
        self.ppm_up3 = nn.Sequential(
            ConvBlock(in_ch=init_filters * 2 ** 3, out_ch=init_filters * 2 ** 2, norm=norm, act=act, ksize=1),
            get_upsample_layer(init_filters * 2 ** 1, out_ch=init_filters * 2 ** 1, upsample_mode=upsample_mode,
                               scale_factor=2),
        )  # 1/8 ->1/2
        self.ppm_up2 = nn.Sequential(
            ConvBlock(in_ch=init_filters * 2 ** 3, out_ch=init_filters * 2 ** 1, norm=norm, act=act, ksize=1),
            get_upsample_layer(init_filters * 2 ** 1, out_ch=init_filters * 2 ** 1, upsample_mode=upsample_mode,
                               scale_factor=4),
        )  # 1/8 ->1/2
        self.ppm_up1 = nn.Sequential(
            ConvBlock(in_ch=init_filters * 2 ** 3, out_ch=init_filters * 2 ** 0, norm=norm, act=act, ksize=1),
            get_upsample_layer(init_filters * 2 ** 0, out_ch=init_filters * 2 ** 0, upsample_mode=upsample_mode,
                               scale_factor=8),
        )  # 1/8 ->1/2

        # FA modules
        # 1/4 -> 1/1
        self.encoder3_up3 = nn.Sequential(
            ConvBlock(in_ch=init_filters * 2 ** 2, out_ch=init_filters, norm=norm, act=act, ksize=1),
            get_upsample_layer(init_filters, out_ch=init_filters, upsample_mode=upsample_mode,
                               scale_factor=4),
        )
        self.encoder2_up2 = nn.Sequential(
            ConvBlock(in_ch=init_filters * 2 ** 1, out_ch=init_filters, norm=norm, act=act,ksize=1),
            get_upsample_layer(init_filters, out_ch=init_filters, upsample_mode=upsample_mode,
                               scale_factor=2),
        )

        self.FA_fusion = ConvBlock(in_ch=init_filters, out_ch=init_filters, norm=norm, act=act)

    def forward(self, x):
        # encoder process
        encoder1 = self.encoder1(x)  # 1/1

        dp1 = self.dp1(encoder1)
        encoder2 = self.encoder2(encoder1)  # 1/2
        encoder2 = dp1 + encoder2
        encoder2 = self.dp1_fusion(encoder2)

        dp2 = self.dp2(encoder1)
        encoder3 = self.encoder3(encoder2)  # 1/4
        encoder3 = dp2 + encoder3
        encoder3 = self.dp2_fusion(encoder3)

        dp3 = self.dp3(encoder1)
        encoder4 = self.encoder4(encoder3)  # 1/8
        encoder4 = dp3 + encoder4
        encoder4 = self.dp3_fusion(encoder4)

        encoder_ppm = self.ppm(encoder4)

        # decoder
        decoder_up3 = self.up3(encoder4) + encoder3 + self.ppm_up3(encoder_ppm)
        decoder_up3 = self.up_fusion3(decoder_up3)

        decoder_up2 = self.up2(decoder_up3) + encoder2 + self.ppm_up2(encoder_ppm)
        decoder_up2 = self.up_fusion2(decoder_up2)

        decoder_up1 = self.up1(decoder_up2) + encoder1 + self.ppm_up1(encoder_ppm)
        decoder_up1 = self.up_fusion1(decoder_up1)

        # FA 结构
        FA3 = self.encoder3_up3(decoder_up3)
        FA2 = self.encoder2_up2(decoder_up2)

        FAFusion = self.FA_fusion(FA2 + FA3 + decoder_up1)

        out = self.conv_final(FAFusion)

        # 采用deep supervision 的方法进行处理

        return out


if __name__ == '__main__':
    import torch
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FRNet(init_filters=32)
    model = net.to(device)

    summary(model, (1, 128, 128, 128))
