# 下采样过程，三次下采样，全部使用长条池化，池化核为（4,2,2）
import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            # nn.MaxPool3d(kernel_size=(2, 4, 4), stride=(2, 4, 4), padding=(0, 1, 1)),
            nn.MaxPool3d(kernel_size=(4, 2, 2), stride=(4, 2, 2), padding=(1, 0, 0)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # if bilinear:
        self.up = nn.Upsample(scale_factor=(4, 2, 2), mode='trilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
        # else:
        #     self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        #     self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2,
                                    diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class LongPool_All(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(LongPool_All, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # self.bilinear = bilinear

        # v1
        # self.inc = DoubleConv(n_channels, 64)
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # factor = 2 if bilinear else 1
        # self.down3 = Down(256, 512 // factor)
        #
        # self.up2 = Up(512, 256 // factor, bilinear)
        # self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes)

        # v2
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)

        self.up2 = Up(192, 64)
        self.up3 = Up(96, 32)
        self.up4 = Up(48, 16)
        self.outc = OutConv(16, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder部分
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # decoder部分
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        outputs = self.softmax(logits)
        return outputs


if __name__ == '__main__':
    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = LongPool_All(1, 2).to(device)
    summary(net, input_size=(1, 128, 128, 128))

# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv3d-1    [-1, 64, 128, 128, 128]           1,792
#        BatchNorm3d-2    [-1, 64, 128, 128, 128]             128
#               ReLU-3    [-1, 64, 128, 128, 128]               0
#             Conv3d-4    [-1, 64, 128, 128, 128]         110,656
#        BatchNorm3d-5    [-1, 64, 128, 128, 128]             128
#               ReLU-6    [-1, 64, 128, 128, 128]               0
#         DoubleConv-7    [-1, 64, 128, 128, 128]               0
#          MaxPool3d-8       [-1, 64, 64, 32, 32]               0
#             Conv3d-9      [-1, 128, 64, 32, 32]         221,312
#       BatchNorm3d-10      [-1, 128, 64, 32, 32]             256
#              ReLU-11      [-1, 128, 64, 32, 32]               0
#            Conv3d-12      [-1, 128, 64, 32, 32]         442,496
#       BatchNorm3d-13      [-1, 128, 64, 32, 32]             256
#              ReLU-14      [-1, 128, 64, 32, 32]               0
#        DoubleConv-15      [-1, 128, 64, 32, 32]               0
#              Down-16      [-1, 128, 64, 32, 32]               0
#         MaxPool3d-17        [-1, 128, 32, 8, 8]               0
#            Conv3d-18        [-1, 256, 32, 8, 8]         884,992
#       BatchNorm3d-19        [-1, 256, 32, 8, 8]             512
#              ReLU-20        [-1, 256, 32, 8, 8]               0
#            Conv3d-21        [-1, 256, 32, 8, 8]       1,769,728
#       BatchNorm3d-22        [-1, 256, 32, 8, 8]             512
#              ReLU-23        [-1, 256, 32, 8, 8]               0
#        DoubleConv-24        [-1, 256, 32, 8, 8]               0
#              Down-25        [-1, 256, 32, 8, 8]               0
#         MaxPool3d-26        [-1, 256, 16, 2, 2]               0
#            Conv3d-27        [-1, 256, 16, 2, 2]       1,769,728
#       BatchNorm3d-28        [-1, 256, 16, 2, 2]             512
#              ReLU-29        [-1, 256, 16, 2, 2]               0
#            Conv3d-30        [-1, 256, 16, 2, 2]       1,769,728
#       BatchNorm3d-31        [-1, 256, 16, 2, 2]             512
#              ReLU-32        [-1, 256, 16, 2, 2]               0
#        DoubleConv-33        [-1, 256, 16, 2, 2]               0
#              Down-34        [-1, 256, 16, 2, 2]               0
#          Upsample-35        [-1, 256, 32, 8, 8]               0
#            Conv3d-36        [-1, 256, 32, 8, 8]       3,539,200
#       BatchNorm3d-37        [-1, 256, 32, 8, 8]             512
#              ReLU-38        [-1, 256, 32, 8, 8]               0
#            Conv3d-39        [-1, 128, 32, 8, 8]         884,864
#       BatchNorm3d-40        [-1, 128, 32, 8, 8]             256
#              ReLU-41        [-1, 128, 32, 8, 8]               0
#        DoubleConv-42        [-1, 128, 32, 8, 8]               0
#                Up-43        [-1, 128, 32, 8, 8]               0
#          Upsample-44      [-1, 128, 64, 32, 32]               0
#            Conv3d-45      [-1, 128, 64, 32, 32]         884,864
#       BatchNorm3d-46      [-1, 128, 64, 32, 32]             256
#              ReLU-47      [-1, 128, 64, 32, 32]               0
#            Conv3d-48       [-1, 64, 64, 32, 32]         221,248
#       BatchNorm3d-49       [-1, 64, 64, 32, 32]             128
#              ReLU-50       [-1, 64, 64, 32, 32]               0
#        DoubleConv-51       [-1, 64, 64, 32, 32]               0
#                Up-52       [-1, 64, 64, 32, 32]               0
#          Upsample-53    [-1, 64, 128, 128, 128]               0
#            Conv3d-54    [-1, 64, 128, 128, 128]         221,248
#       BatchNorm3d-55    [-1, 64, 128, 128, 128]             128
#              ReLU-56    [-1, 64, 128, 128, 128]               0
#            Conv3d-57    [-1, 64, 128, 128, 128]         110,656
#       BatchNorm3d-58    [-1, 64, 128, 128, 128]             128
#              ReLU-59    [-1, 64, 128, 128, 128]               0
#        DoubleConv-60    [-1, 64, 128, 128, 128]               0
#                Up-61    [-1, 64, 128, 128, 128]               0
#            Conv3d-62     [-1, 2, 128, 128, 128]             130
#           OutConv-63     [-1, 2, 128, 128, 128]               0
# ================================================================
# Total params: 12,836,866
# Trainable params: 12,836,866
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 8.00
# Forward/backward pass size (MB): 17469.12
# Params size (MB): 48.97
# Estimated Total Size (MB): 17526.09
# ----------------------------------------------------------------
