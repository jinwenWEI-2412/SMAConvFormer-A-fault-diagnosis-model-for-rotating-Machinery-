import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Mish(nn.Module):
    """Mish激活函数实现"""

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class WaveletWeightInitializer:
    """小波权重初始化器，用于初始化卷积核为特定小波函数"""

    def __init__(self, wavelet_type: str = 'laplace', scale: float = 1.0):
        self.wavelet_type = wavelet_type
        self.scale = scale

    def __call__(self, out_channels: int, in_channels: int, kernel_size: int) -> torch.Tensor:
        """生成小波初始化的卷积核权重"""
        x = torch.linspace(-self.scale, self.scale, kernel_size)
        weights = torch.zeros(out_channels, in_channels, kernel_size)

        for c in range(in_channels):
            for i in range(out_channels):
                # 调整每个通道的尺度参数，使其具有多样性
                channel_scale = self.scale * (1.0 + 0.1 * (i % 10))

                if self.wavelet_type == 'laplace':
                    # 拉普拉斯小波
                    wavelet = torch.exp(-torch.abs(x / channel_scale)) * torch.sin(2 * torch.pi * x / channel_scale)
                elif self.wavelet_type == 'morlet':
                    # Morlet小波
                    wavelet = torch.exp(-0.5 * (x / channel_scale) ** 2) * torch.cos(5 * x / channel_scale)
                else:
                    # 默认使用高斯导数小波
                    wavelet = -x / channel_scale * torch.exp(-0.5 * (x / channel_scale) ** 2)

                weights[i, c, :] = wavelet

        # 归一化权重
        return weights / torch.norm(weights, p=2, dim=2, keepdim=True)


class BalancedDynamicThreshold(nn.Module):
    """平衡动态自适应阈值模块，用于降噪"""

    def __init__(self, channel: int, reduction: int = 16):
        super(BalancedDynamicThreshold, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            Mish(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # 可学习的阈值平衡参数
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)

        # 平衡硬阈值和软阈值
        threshold = y * self.alpha

        # 应用阈值
        mask = torch.abs(x) > threshold
        return x * mask * (1 - threshold / (torch.abs(x) + 1e-8))


class EWSNet(nn.Module):
    """EWSNet: 用于滚动轴承故障诊断的三流网络"""

    def __init__(self, args, in_channel, out_channel, wavelet_type: str = 'laplace'):
        """
        初始化EWSNet模型

        参数:
            args: 命令行参数
            in_channel: 输入通道数
            out_channel: 输出类别数
            wavelet_type: 小波类型，默认为'laplace'
        """
        super(EWSNet, self).__init__()
        self.args = args  # 保存参数，可能在其他地方需要使用

        # 使用传入的参数
        self.in_channels = in_channel
        self.num_classes = out_channel

        # 初始化小波权重生成器
        self.wavelet_initializer = WaveletWeightInitializer(wavelet_type=wavelet_type)

        # 第一流：低频特征提取
        self.stream1 = nn.Sequential(
            self._create_wavelet_conv(in_channel, 64, kernel_size=250, stride=1),
            nn.BatchNorm1d(64),
            Mish(),
            nn.Conv1d(64, 16, kernel_size=18, stride=2, bias=True),
            nn.BatchNorm1d(16),
            Mish(),
            nn.Conv1d(16, 10, kernel_size=10, stride=2, bias=True),
            nn.BatchNorm1d(10),
            Mish(),
            nn.MaxPool1d(kernel_size=2)
        )

        # 第二流：中频特征提取
        self.stream2 = nn.Sequential(
            self._create_wavelet_conv(in_channel, 64, kernel_size=100, stride=1),
            nn.BatchNorm1d(64),
            Mish(),
            nn.Conv1d(64, 32, kernel_size=6, stride=1, bias=True),
            nn.BatchNorm1d(32),
            Mish(),
            nn.Conv1d(32, 16, kernel_size=6, stride=1, bias=True),
            nn.BatchNorm1d(16),
            Mish(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 10, kernel_size=6, stride=1, bias=True),
            nn.BatchNorm1d(10),
            Mish(),
            nn.Conv1d(10, 10, kernel_size=8, stride=2, bias=True),
            nn.BatchNorm1d(10),
            Mish(),
            nn.MaxPool1d(kernel_size=2)
        )

        # 第三流：高频特征提取与降噪
        self.stream3 = nn.Sequential(
            self._create_wavelet_conv(in_channel, 64, kernel_size=50, stride=1),
            nn.BatchNorm1d(64),
            Mish(),
            BalancedDynamicThreshold(64),  # 自适应阈值降噪
            nn.Conv1d(64, 10, kernel_size=15, stride=2, bias=True),
            nn.BatchNorm1d(10),
            Mish(),
            nn.MaxPool1d(kernel_size=2)
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv1d(30, 30, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(30),
            Mish(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 分类器 - 使用out_channel
        self.classifier = nn.Sequential(
            nn.Linear(30, 64),
            Mish(),
            nn.Dropout(0.5),
            nn.Linear(64, out_channel)
        )

        # 初始化权重
        self._initialize_other_weights()

    def _create_wavelet_conv(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> nn.Conv1d:
        """创建使用小波初始化的卷积层"""
        conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, bias=True)
        conv.weight.data = self.wavelet_initializer(out_channels, in_channels, kernel_size)
        nn.init.constant_(conv.bias.data, 0.0)
        return conv

    def _initialize_other_weights(self):
        """初始化非小波卷积层的权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) and not hasattr(m, '_wavelet_initialized'):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 三流并行处理
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)



        # 确保所有流的输出在非通道维度上具有相同的尺寸
        target_size = s1.shape[2]  # 获取 s1 的序列长度作为目标尺寸

        # 如果 s2 和 s3 的尺寸与 s1 不同，则进行调整
        if s2.shape[2] != target_size:
            s2 = nn.AdaptiveAvgPool1d(target_size)(s2)

        if s3.shape[2] != target_size:
            s3 = nn.AdaptiveAvgPool1d(target_size)(s3)



        # 特征融合
        fused = torch.cat([s1, s2, s3], dim=1)

        # 通过融合层和分类器
        x = self.fusion(fused).squeeze(-1)
        x = self.classifier(x)

        return x


def ewsnet(**kwargs):
    """创建EWSNet模型的函数"""
    return EWSNet(**kwargs)


if __name__ == '__main__':
    # 创建一个模拟的args对象
    class Args:
        def __init__(self):
            self.wavelet_type = 'laplace'


    # 测试模型 - 修改为兼容项目调用方式
    args = Args()
    model = EWSNet(args, in_channel=3, out_channel=10)
    x = torch.randn(2, 3, 1024)
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")