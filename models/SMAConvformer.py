import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from models.SCSA import SCSA  # Assumes SCSA is imported from your module

class ODConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, reduction=4):
        #groups : 分组卷积的组数，默认为 1；  eduction 通道注意力中的压缩比例（用于减少参数量）
        super(ODConv1d, self).__init__()
        # if out_channels <= 0:
        #     raise ValueError(f"ODConv1d Error: out_channels must be > 0. Got {out_channels}")
        # if in_channels <= 0:
        #     raise ValueError(f"ODConv1d Error: in_channels must be > 0. Got {in_channels}")

        self.kernel_size = kernel_size
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Channel-wise attention
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, max(in_channels // reduction, 1), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(max(in_channels // reduction, 1), out_channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, L = x.shape
        attn = self.attention(x)  # [B, out_channels, 1]
        weight = self.weight.unsqueeze(0) * attn.unsqueeze(2)  # [B, out_c, in_c/groups, k]
        weight = weight.view(B * self.out_channels, self.in_channels // self.groups, self.kernel_size)

        x = x.view(1, B * C, L)
        out = F.conv1d(x, weight, bias=None, stride=self.stride,
                       padding=self.padding, groups=B * self.groups)
        return out.view(B, self.out_channels, -1) + self.bias.view(1, -1, 1)


class ODConvEmbedding(nn.Module):

    def __init__(self, d_in, d_out, stride=2, n=4):
        super(ODConvEmbedding, self).__init__()
        d_hidden = d_out // n
        self.conv1 = nn.Conv1d(d_in, d_hidden, kernel_size=1, stride=1)

        # ODConv1d-based multi-scale branches
        self.sconv = nn.ModuleList([
            ODConv1d(
                d_hidden, d_hidden,
                kernel_size=2 * i + 2 * stride - 1,
                stride=stride,
                padding=stride + i - 1,
                groups=d_hidden  # depthwise
            ) for i in range(n)
        ])

        self.act_bn = nn.Sequential(
            nn.BatchNorm1d(d_out),
            nn.GELU(),
            nn.Dropout(0.1)  # 添加Dropout
        )

    def forward(self, x):
        x = self.conv1(x)
        out = torch.cat([sconv(x) for sconv in self.sconv], dim=1)
        return self.act_bn(out)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.layernorm(x)
        return x.transpose(-1, -2)


class Add(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(Add, self).__init__()
        self.epsilon = epsilon
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w_relu = nn.ReLU()

    def forward(self, x):
        w = self.w_relu(self.w)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)

        return weight[0] * x[0] + weight[1] * x[1]


class BSA_SCSA_Fusion(nn.Module):
    def __init__(self, channels, scsa_heads=4, window_size=1, drop_prob=0.2):
        super(BSA_SCSA_Fusion, self).__init__()
        self.channels = channels

        # BSA 分支
        self.w_tau = nn.Conv1d(channels, 1, kernel_size=1)
        self.w_k = nn.Conv1d(channels, channels, kernel_size=1)
        self.w_v = nn.Conv1d(channels, channels, kernel_size=1)
        self.w_out_bsa = nn.Conv1d(channels, channels, kernel_size=1)

        # SCSA 分支（来自外部模块）
        self.scsa = SCSA(dim=channels, head_num=scsa_heads, window_size=window_size)

        # 可学习融合参数
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

        self.norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(p=drop_prob)

    def forward(self, x):  # x: [B, C, N]
        B, C, N = x.size()
        #print(f"x output shape: {x.shape}")
        # ===== BSA 分支 =====
        tau = F.softmax(self.w_tau(x), dim=-1)        # [B, 1, N]
        k = self.w_k(x)                               # [B, C, N]
        v = F.relu(self.w_v(x))                       # [B, C, N]
        gamma = (tau * k).sum(dim=-1, keepdim=True)   # [B, C, 1]
        out_bsa = (gamma * v).sum(dim=1, keepdim=True)  # [B, 1, N]
        out_bsa = self.w_out_bsa(out_bsa.expand(-1, C, -1))  # [B, C, N]
        #print(f"out_bsa output shape: {out_bsa.shape}")

        # ===== SCSA 分支 =====
        x_2d = x.unsqueeze(-1)  # [B, C, N, 1]
        out_scsa = self.scsa(x_2d).squeeze(-1)  # [B, C, N]
        #print(f"out_scsa output shape: {out_scsa.shape}")

        # ===== 融合输出 =====
        alpha = torch.clamp(self.fusion_weight, 0.0, 1.0)
        out = alpha * out_bsa + (1 - alpha) * out_scsa  # 加权融合
        out = self.drop(out)                            # Dropout 防过拟合
        out = out + x                                   # 残差连接
        out = out.permute(0, 2, 1)                      # [B, N, C]
        out = self.norm(out)
        #print(f"out shape: {out.shape}")
        return out.permute(0, 2, 1)  # [B, C, N]



class BA_FFN_Block(nn.Module):
    def __init__(self,
                 dim,
                 ffn_dim,
                 drop=0.,
                 attn_drop=0.
                 ):
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.add1 = Add()
        self.attn = BSA_SCSA_Fusion(channels=dim, scsa_heads=4, window_size=1)

        self.norm2 = LayerNorm(dim)
        self.add2 = Add()
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),  # Depthwise
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),  # Pointwise
        )

    def forward(self, x):
        x = self.add1([self.attn(self.norm1(x)), x])
        x = self.add2([self.ffn(self.norm2(x)), x])
        return x


class LFEL(nn.Module):
    def __init__(self, d_in, d_out, drop):
        super(LFEL, self).__init__()

        self.embed = ODConvEmbedding(d_in, d_out, stride=2, n=4)
        self.block = BA_FFN_Block(dim=d_out,
                                  ffn_dim=d_out//4,
                                  drop=drop,
                                  attn_drop=drop)

    def forward(self, x):
        x = self.embed(x)
        return self.block(x)


class SMAConvformer(nn.Module):
    def __init__(self, _, in_channel, out_channel, drop=0.1, dim=32):
        super(SMAConvformer, self).__init__()

        self.in_layer = nn.Sequential(
            nn.AvgPool1d(2, 2),
            ODConv1d(in_channel, dim, kernel_size=15, stride=2)
        )

        self.LFELs = nn.Sequential(
            LFEL(dim, 2*dim, drop),
            LFEL(2*dim, 4*dim, drop),
            LFEL(4*dim, 8*dim, drop),
            nn.AdaptiveAvgPool1d(1)
        )

        self.out_layer = nn.Linear(8*dim, out_channel)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.LFELs(x)
        x = self.out_layer(x.squeeze())
        return x


