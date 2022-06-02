import torch
import torch.nn as nn
import torch.nn.functional as F


class SEnet(nn.Module):
    def __init__(self, in_channel, scale):
        super(SEnet, self).__init__()
        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
            nn.Linear(in_channel, in_channel // scale),
            nn.ReLU(),
            nn.Linear(in_channel // scale, in_channel)
        )
        self.activate = nn.Sigmoid()

    def forward(self, x):
        """
        @params: x (B,C,H,W)

        @return: out (B,C)
        """
        x = self.linear(self.avgPool(x))
        out = self.activate(x)
        return out


class CBAM(nn.Module):
    class ChannelAttentionModule(nn.Module):
        def __init__(self, in_channel, scale):
            super(CBAM.ChannelAttentionModule, self).__init__()
            self.maxPool = nn.AdaptiveMaxPool2d(1)
            self.avgPool = nn.AdaptiveAvgPool2d(1)

            self.linear = nn.Sequential(
                nn.Linear(in_channel, in_channel // scale),
                nn.ReLU(),
                nn.Linear(in_channel // scale, in_channel)
            )

            self.subAttention = nn.Conv1d(2, 1, 1)
            self.activate = nn.Sigmoid()

        def forward(self, x):
            """
            @params: x (B,C,H,W)

            @return: out (B,C,H,W)
            """
            batch, channel, height, width = x.shape
            max_feat = self.linear(self.maxPool(x))  # (B,C)
            avg_feat = self.linear(self.avgPool(x))  # (B,C)

            attention_feat = torch.stack((max_feat, avg_feat), dim=1)  # (B,2,C)
            attention_feat = self.subAttention(attention_feat).squeeze(1)  # (B,C)

            out = self.activate(attention_feat)
            out = x * out.view(batch, channel, 1, 1)
            return out

    class SpatialAttentionModule(nn.Module):
        def __init__(self, in_channel, kernel_size=7):
            """
            @params: kernel_size, should be odd
            """
            super(CBAM.SpatialAttentionModule, self).__init__()
            assert kernel_size % 2 == 1
            self.padding = kernel_size // 2
            self.conv2 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=self.padding)
            self.activate = nn.Sigmoid()

        def forward(self, x):
            """
            @params: x, (B,C,H,W)

            @return: out, (B,C,H,W)
            """
            batch, channel, height, width = x.shape
            max_feat = torch.max(x, dim=1, keepdim=True)
            avg_feat = torch.mean(x, dim=1, keepdim=True)

            attention_feat = torch.cat([max_feat, avg_feat], dim=1)
            attention_feat = self.conv2(attention_feat)

            out = self.activate(attention_feat)  # (B,H,W)
            out = x * out.view(batch, 1, height, width)
            return out

    def __init__(self, in_channel, channel_scale=16, spatial_kernel=7, mode=0):
        super().__init__()
        self.channelFeat = CBAM.ChannelAttentionModule(in_channel=in_channel, scale=channel_scale)
        self.spatialFeat = CBAM.SpatialAttentionModule(in_channel=in_channel, kernel_size=spatial_kernel)

        self.backbone = nn.Sequential()
        if mode == 0:
            self.attentionFeat.add_module("channel", self.channelFeat)
            self.attentionFeat.add_module("spatial", self.spatialFeat)
        elif mode == 1:
            self.attentionFeat.add_module("spatial", self.spatialFeat)
            self.attentionFeat.add_module("channel", self.channelFeat)

    def forward(self, x):
        """
        @params: x, (B,C,H,W)
        """
        x = self.backbone(x)
        return x


class ECAnet(nn.Module):
    def __init__(self, in_channel):
        super(ECAnet, self).__init__()
        pass
