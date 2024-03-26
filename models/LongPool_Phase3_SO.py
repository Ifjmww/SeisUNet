# 下采样过程，仅第三次下采样使用长条池化，池化核为（4,2,2），前两次使用最大池化，并添加侧输出（SideOutput,SO）


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
from torchsummaryX import summary
from torchsummary import summary


def _upsample_like_3D(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode='trilinear')

    return src


class PreProcess3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PreProcess3D, self).__init__()
        self.preprocess_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.preprocess_conv(x)


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class LongPool_Down3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LongPool_Down3D, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(4, 2, 2), stride=(4, 2, 2), padding=(1, 0, 0)),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Down3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down3D, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class LongPool_Up3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LongPool_Up3D, self).__init__()

        self.up = nn.Upsample(scale_factor=(4, 2, 2), mode='trilinear', align_corners=True)
        self.conv = DoubleConv3D(in_channels + out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Up3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up3D, self).__init__()

        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        self.conv = DoubleConv3D(in_channels + out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class LongPool_Phase3_SO(nn.Module):
    """
    UNet3D_LongPool_SideOutput

    """

    def __init__(self, in_channels, out_channels):
        super(LongPool_Phase3_SO, self).__init__()

        # encoder
        self.down1 = PreProcess3D(in_channels, 32)
        self.down2 = Down3D(32, 64)
        self.down3 = Down3D(64, 128)
        self.down4 = LongPool_Down3D(128, 256)

        # decoder
        self.up1 = LongPool_Up3D(256, 128)
        self.up2 = Up3D(128, 64)
        self.up3 = Up3D(64, 32)

        # side output
        self.side_enc_4 = nn.Conv3d(256, out_channels, kernel_size=1)
        self.side_dec_3 = nn.Conv3d(128, out_channels, kernel_size=1)
        self.side_dec_2 = nn.Conv3d(64, out_channels, kernel_size=1)

        # out logits
        self.outconv = OutConv3D(32, out_channels)

        # out Probability
        self.output = nn.Sequential(
            nn.Conv3d(4 * out_channels, out_channels, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # encoder
        enc_1 = self.down1(x)
        enc_2 = self.down2(enc_1)
        enc_3 = self.down3(enc_2)
        enc_4 = self.down4(enc_3)

        # decoder
        dec_3 = self.up1(enc_4, enc_3)
        dec_2 = self.up2(dec_3, enc_2)
        dec_1 = self.up3(dec_2, enc_1)

        # side output
        side_enc_4 = _upsample_like_3D(self.side_enc_4(enc_4), x)
        side_dec_3 = _upsample_like_3D(self.side_dec_3(dec_3), x)
        side_dec_2 = _upsample_like_3D(self.side_dec_2(dec_2), x)

        # out logits
        logits = self.outconv(dec_1)

        # out Probability
        outputs = self.output(torch.cat((logits, side_dec_2, side_dec_3, side_enc_4), 1))

        return outputs


if __name__ == '__main__':
    # 定义网络
    model = LongPool_Phase3_SO(3, 2).to('cuda:0')

    # 模拟输入数据
    input_data = torch.zeros((1, 3, 128, 128, 128)).to('cuda:0')

    # 使用torchviz生成感受野的热图
    # output = model(input_data)
    # summary(model, input_size=(3, 128, 128, 128))
    print(model)