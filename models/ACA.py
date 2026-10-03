import torch
import torch.nn as nn
import torch.nn.functional as F

#------------------------------------默认 ACA start----------------------------------------------------
# # SimAM:SimAM（Simple Attention Module）注意力模块，它是一种轻量级、无参数的注意力机制
class SimAM(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activation(y)

# ELA 模块
class ELA(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.GroupNorm(16, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.conv1x1(self.pool_h(x).reshape((b, c, h))).reshape((b, c, h, 1))
        x_w = self.conv1x1(self.pool_w(x).reshape((b, c, w))).reshape((b, c, 1, w))
        return x * x_h * x_w

# # 顺序串联 SimAM 和 ELA 模块
# class SimAM_ELA(nn.Module):
#     def __init__(self, channels, e_lambda=1e-4):
#         super(SimAM_ELA, self).__init__()
#         self.simam = SimAM(channels, e_lambda)
#         self.ela = ELA(channels)

#     def forward(self, x):
#         # 先通过 SimAM 模块
#         x = self.simam(x)
#         # 再通过 ELA 模块
#         x = self.ela(x)
#         # print(f'SimAM_ELA size: {x.size()}')
#         return x
#------------------------------------默认 ACA end----------------------------------------------------

#------------------------------------ACA start-----------------------------------------------------
# 1.减少计算复杂度:在 SimAM 模块中，计算方差时需要考虑避免重复操作，可以通过预先计算均值来减少重复计算。
# 2.通道数的动态调整:在 ELA 模块中，可以根据输入的通道数动态调整 GroupNorm 的参数，以适应不同的网络结构。
# 3.减少不必要的重塑操作:在 ELA 模块的 forward 方法中，尽量避免多次reshape操作，以提高效率。(效果不好)
# 4.使用残差连接:在 ACA 的 forward 方法中，考虑加入残差连接，以帮助模型在深层网络中更好地传播梯度。（效果不好）

class SimAMv1(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAMv1, self).__init__()
        self.activation = nn.Sigmoid()  
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1

        mu = x.mean(dim=[2, 3], keepdim=True)
        x_minus_mu_square = (x - mu).pow(2)
        variance = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda
        y = x_minus_mu_square / (4 * variance) + 0.5

        return x * self.activation(y)

# SimAMv2 动态调整 e_lambda
class SimAMv2(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAMv2, self).__init__()
        self.activation = nn.Sigmoid()
        # 将 e_lambda 定义为可学习参数
        self.e_lambda = nn.Parameter(torch.tensor(e_lambda, dtype=torch.float32))

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda.item())
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1

        mu = x.mean(dim=[2, 3], keepdim=True)
        x_minus_mu_square = (x - mu).pow(2)
        variance = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda
        y = x_minus_mu_square / (4 * variance) + 0.5

        return x * self.activation(y)

class ELAv1(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.GroupNorm(min(16, channels), channels),  # 动态调整
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.conv1x1(self.pool_h(x).flatten(2)).reshape((b, c, h, 1)) # 避免多次reshape操作，以提高效率
        x_w = self.conv1x1(self.pool_w(x).flatten(2)).reshape((b, c, 1, w))
        return x * x_h * x_w

# ELAv2引入深度可分离卷积
class ELAv2(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        # 深度可分离卷积
        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size=1, groups=channels)
        self.pointwise_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv1x1 = nn.Sequential(
            self.depthwise_conv,
            self.pointwise_conv,
            nn.GroupNorm(16, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.conv1x1(self.pool_h(x).reshape((b, c, h))).reshape((b, c, h, 1))
        x_w = self.conv1x1(self.pool_w(x).reshape((b, c, w))).reshape((b, c, 1, w))
        return x * x_h * x_w

class ACA(nn.Module):
    def __init__(self, channels, e_lambda=1e-4):
        super(ACA, self).__init__()
        self.simam = SimAMv1(channels, e_lambda)
        self.ela = ELAv1(channels)

    def forward(self, x):
        residual = x  # 保存输入以便于残差连接
        x = self.simam(x)
        x = self.ela(x)
        return x + residual  # 残差连接
    
class ACAv3(nn.Module):
    def __init__(self, channels, e_lambda=1e-4):
        super(ACAv3, self).__init__()
        self.simam = SimAMv2(channels, e_lambda)
        self.ela = ELA(channels)

    def forward(self, x):
        # 先通过 SimAM 模块
        x = self.simam(x)
        # 再通过 ELA 模块
        x = self.ela(x)
        # print(f'SimAM_ELA size: {x.size()}')
        return x
    
class ACAv4(nn.Module):
    def __init__(self, channels, e_lambda=1e-4):
        super(ACAv4, self).__init__()
        self.simam = SimAM(channels, e_lambda)
        self.ela = ELAv2(channels)

    def forward(self, x):
        # 先通过 SimAM 模块
        x = self.simam(x)
        # 再通过 ELA 模块
        x = self.ela(x)
        # print(f'SimAM_ELA size: {x.size()}')
        return x
#------------------------------------ACA end-----------------------------------------------------

#------------------------------------ACAv2 start-------------------------------------------------
# 并行加权+残差连接  
class ACAv2(nn.Module):
    def __init__(self, channels, e_lambda=1e-4):
        super(ACAv2, self).__init__()
        self.simam = SimAMv1(channels, e_lambda)
        self.ela = ELAv1(channels)
        # 可学习的权重
        self.weight_simam = nn.Parameter(torch.ones(1))
        self.weight_ela = nn.Parameter(torch.ones(1))

    def forward(self, x):
        residual = x  # 保存输入以便于残差连接
        simam_out = self.simam(x)
        ela_out = self.ela(x)
        weighted_simam = self.weight_simam * simam_out
        weighted_ela = self.weight_ela * ela_out
        merged_out = weighted_simam + weighted_ela
        # print(f'weighted_simam size: {weighted_simam.size()}')

        return merged_out+residual  # 残差连接
#------------------------------------ACAv2 end-------------------------------------------------