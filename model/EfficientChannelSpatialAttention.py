import torch
from torch import nn
import math
import torch.nn.functional as F
from config import cfg

class ecsa_layer(nn.Module):
    def __init__(self, channels, k_size=3, channel_attention=cfg['channel_attention'], spatial_attention=cfg['spatial_attention']):
        super(ecsa_layer, self).__init__()
        self.channel_attention = channel_attention
        self.spatial_attention = spatial_attention
        if k_size <= 0:
            k_size = int((math.log2(channels)+5)/2)
            k_size = k_size if k_size % 2 else k_size +1 # int(k_size + math.fmod(k_size, 2) - 1)
            #print(k_size)
        if channel_attention:
            self.gap_c = nn.AdaptiveAvgPool2d((channels, 1))
            self.gmp_c = nn.AdaptiveMaxPool2d((channels, 1))
            self.conv_c = nn.Conv1d(2, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        if spatial_attention:
            self.gap_s = nn.AdaptiveAvgPool2d((1, None))
            self.gmp_s = nn.AdaptiveMaxPool2d((1, None))
            self.conv_s = nn.Conv1d(2, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.channel_attention:
            y1_c = self.gap_c(x)
            y2_c = self.gmp_c(x)
            y = torch.cat([y1_c, y2_c], dim=-1)
            y = y.transpose(-1, -2)
            y = self.conv_c(y)
            y = self.sigmoid(y)
            channel_att_map = y.transpose(-1, -2)
            x1 = x * channel_att_map
        else:
            x1 = x
            channel_att_map = torch.ones_like(torch.sum(x, dim=-1, keepdim=True))
            y1_c = torch.zeros_like(channel_att_map)
            y2_c = torch.zeros_like(channel_att_map)

        if self.spatial_attention:
            y1_s = self.gap_s(x)
            y2_s = self.gmp_s(x)
            y = torch.cat([y1_s, y2_s], dim=-2)
            y = self.conv_s(y)
            spatial_att_map = self.sigmoid(y)
            x2 = x * spatial_att_map
        else:
            x2 = x
            spatial_att_map = torch.ones_like(torch.sum(x, dim=-2, keepdim=True))
            y1_s = torch.zeros_like(spatial_att_map)
            y2_s = torch.zeros_like(spatial_att_map)
        return x1 + x2, (channel_att_map, spatial_att_map, y1_c, y2_c, y1_s, y2_s)




