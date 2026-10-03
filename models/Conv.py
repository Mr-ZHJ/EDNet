import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------SPD-Conv-----------------------------
 
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class SPDConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        c1 = c1 * 4
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.act(self.conv(x))
# --------------------------------SPD-Conv-----------------------------

# --------------------------------PConv-----------------------------
# class Conv(nn.Module):
#     # Standard convolution
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))

#     def forward_fuse(self, x):
#         return self.act(self.conv(x))

class PConv(nn.Module):
    def __init__(self, dim, ouc, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.conv = Conv(dim, ouc, k=1)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        x = self.conv(x)
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        x = self.conv(x)
        return x

# --------------------------------PConv-----------------------------


# # --------------------------------DCNConv-----------------------------

# from torchvision.ops import DeformConv2d
# class DCNConv(nn.Module):
#     # Standard convolution
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super().__init__()
#         self.conv1 = nn.Conv2d(c1, c2, 3, 2, 1, groups=g, bias=False)
#         deformable_groups = 1
#         offset_channels = 18
#         self.conv2_offset = nn.Conv2d(c2, deformable_groups * offset_channels, kernel_size=3, padding=1)
#         self.conv2 = DeformConv2d(c2, c2, kernel_size=3, padding=1, bias=False)

#         # self.conv2 = DeformableConv2d(c2, c2, k, s, autopad(k, p), groups=g, bias=False)
#         self.bn1 = nn.BatchNorm2d(c2)
#         self.act1 = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
#         self.bn2 = nn.BatchNorm2d(c2)
#         self.act2 = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

#     def forward(self, x):
#         # print(x.shape)
#         # print('-'*50)
#         x = self.act1(self.bn1(self.conv1(x)))
#         # print(x.shape)
#         offset = self.conv2_offset(x)
#         x = self.act2(self.bn2(self.conv2(x, offset)))
#         # print('-'*50)
#         # print(x.shape)
#         return x

# # --------------------------------DCNConv-----------------------------

#-----------------------------------------Dynamic_conv2d Start-------------------------------------------
 
class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=False)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()
 
 
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
 
    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))
 
 
    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)
 
 
class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)
 
        self.weight = nn.Parameter(torch.Tensor(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()
 
        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])
 
 
    def update_temperature(self):
        self.attention.updata_temperature()
 
    def forward(self, x): # 将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)
 
        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
 
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output
# -----------------------------------------Dynamic_conv2d End-------------------------------------------

######################################## LAWDS begin ########################################
from einops import rearrange

class LAWDS(nn.Module):
    # Light Adaptive-weight downsampling
    def __init__(self, ch, group=16) -> None:
        super().__init__()
        
        self.softmax = nn.Softmax(dim=-1)
        self.attention = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(ch, ch, k=1)
        )
        
        self.ds_conv = Conv(ch, ch * 4, k=3, s=2, g=(ch // group))
        
    
    def forward(self, x):
        # bs, ch, 2*h, 2*w => bs, ch, h, w, 4
        att = rearrange(self.attention(x), 'bs ch (s1 h) (s2 w) -> bs ch h w (s1 s2)', s1=2, s2=2)
        att = self.softmax(att)
        
        # bs, 4 * ch, h, w => bs, ch, h, w, 4
        x = rearrange(self.ds_conv(x), 'bs (s ch) h w -> bs ch h w s', s=4)
        x = torch.sum(x * att, dim=-1)
        return x
    
######################################## LAWDS end ########################################

######################################## ContextGuidedBlock start ########################################

class FGlo(nn.Module):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
    """
    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ContextGuidedBlock(nn.Module):
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=True):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, 
           add: if true, residual learning
        """
        super().__init__()
        n= int(nOut/2)
        self.conv1x1 = Conv(nIn, n, 1, 1)  #1x1 Conv is employed to reduce the computation
        self.F_loc = nn.Conv2d(n, n, 3, padding=1, groups=n)
        self.F_sur = nn.Conv2d(n, n, 3, padding=autopad(3, None, dilation_rate), dilation=dilation_rate, groups=n) # surrounding context
        self.bn_act = nn.Sequential(
            nn.BatchNorm2d(nOut),
            Conv.default_act
        )
        self.add = add
        self.F_glo= FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        
        joi_feat = torch.cat([loc, sur], 1) 

        joi_feat = self.bn_act(joi_feat)

        output = self.F_glo(joi_feat)  #F_glo is employed to refine the joint feature
        # if residual version
        if self.add:
            output  = input + output
        return output

class ContextGuidedBlock_Down(nn.Module):
    """
    the size of feature map divided 2, (H,W,C)---->(H/2, W/2, 2C)
    """
    # default_act = nn.SiLU()  # default activation
    
    def __init__(self, nIn, dilation_rate=2, reduction=16):
        """
        args:
           nIn: the channel of input feature map
           nOut: the channel of output feature map, and nOut=2*nIn
        """
        super().__init__()
        nOut = 2 * nIn
        self.conv1x1 = Conv(nIn, nOut, 3, s=2)  #  size/2, channel: nIn--->nOut
        
        self.F_loc = nn.Conv2d(nOut, nOut, 3, padding=1, groups=nOut)
        self.F_sur = nn.Conv2d(nOut, nOut, 3, padding=autopad(3, None, dilation_rate), dilation=dilation_rate, groups=nOut) 
        
        self.bn = nn.BatchNorm2d(2 * nOut, eps=1e-3)
        # self.act = Conv.default_act
        self.act =  nn.SiLU()  # default activation
        self.reduce = Conv(2 * nOut, nOut,1,1)  #reduce dimension: 2*nOut--->nOut
        
        self.F_glo = FGlo(nOut, reduction)    

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)

        joi_feat = torch.cat([loc, sur],1)  #  the joint feature
        joi_feat = self.bn(joi_feat)
        joi_feat = self.act(joi_feat)
        joi_feat = self.reduce(joi_feat)     #channel= nOut
        
        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature
        # print("output shape:", output.shape)
        return output

# class C3_ContextGuided(C3):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.m = nn.Sequential(*(ContextGuidedBlock(c_, c_) for _ in range(n)))

# class C2f_ContextGuided(C2f):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.m = nn.ModuleList(ContextGuidedBlock(self.c, self.c) for _ in range(n))

######################################## ContextGuidedBlock end ########################################

######################################## HWD start ########################################
# 需要安装pytorch_wavelets
# 需要关闭AMP
# class HWD(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(HWD, self).__init__()
#         from pytorch_wavelets import DWTForward
#         self.wt = DWTForward(J=1, mode='zero', wave='haar')
#         self.conv = Conv(in_ch * 4, out_ch, 1, 1)
         
#     def forward(self, x):
#         yL, yH = self.wt(x)
#         y_HL = yH[0][:,:,0,::]
#         y_LH = yH[0][:,:,1,::]
#         y_HH = yH[0][:,:,2,::]
#         x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)        
#         x = self.conv(x)

#         return x

######################################## HWD end ########################################

#------------------------------------------SAConv2d start-----------------------------------
class ConvAWS2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.register_buffer('weight_gamma', torch.ones(self.out_channels, 1, 1, 1))
        self.register_buffer('weight_beta', torch.zeros(self.out_channels, 1, 1, 1))
 
    def _get_weight(self, weight):
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(weight.view(weight.size(0), -1).var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        weight = weight / std
        weight = self.weight_gamma * weight + self.weight_beta
        return weight
 
    def forward(self, x):
        weight = self._get_weight(self.weight)
        return super()._conv_forward(x, weight, None)
 
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.weight_gamma.data.fill_(-1)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        if self.weight_gamma.data.mean() > 0:
            return
        weight = self.weight.data
        weight_mean = weight.data.mean(dim=1, keepdim=True).mean(dim=2,
                                                                 keepdim=True).mean(dim=3, keepdim=True)
        self.weight_beta.data.copy_(weight_mean)
        std = torch.sqrt(weight.view(weight.size(0), -1).var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        self.weight_gamma.data.copy_(std)
 
 
class SAConv2d(ConvAWS2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 s=1,
                 p=None,
                 g=1,
                 d=1,
                 act=True,
                 bias=True):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=s,
            padding=autopad(kernel_size, p),
            dilation=d,
            groups=g,
            bias=bias)
        self.switch = torch.nn.Conv2d(
            self.in_channels,
            1,
            kernel_size=1,
            stride=s,
            bias=True)
        self.switch.weight.data.fill_(0)
        self.switch.bias.data.fill_(1)
        self.weight_diff = torch.nn.Parameter(torch.Tensor(self.weight.size()))
        self.weight_diff.data.zero_()
        self.pre_context = torch.nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=1,
            bias=True)
        self.pre_context.weight.data.fill_(0)
        self.pre_context.bias.data.fill_(0)
        self.post_context = torch.nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=1,
            bias=True)
        self.post_context.weight.data.fill_(0)
        self.post_context.bias.data.fill_(0)
 
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
 
    def forward(self, x):
        # pre-context
        avg_x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)
        avg_x = self.pre_context(avg_x)
        avg_x = avg_x.expand_as(x)
        x = x + avg_x
        # switch
        avg_x = torch.nn.functional.pad(x, pad=(2, 2, 2, 2), mode="reflect")
        avg_x = torch.nn.functional.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)
        switch = self.switch(avg_x)
        # sac
        weight = self._get_weight(self.weight)
        out_s = super()._conv_forward(x, weight, None)
        ori_p = self.padding
        ori_d = self.dilation
        self.padding = tuple(3 * p for p in self.padding)
        self.dilation = tuple(3 * d for d in self.dilation)
        weight = weight + self.weight_diff
        out_l = super()._conv_forward(x, weight, None)
        out = switch * out_s + (1 - switch) * out_l
        self.padding = ori_p
        self.dilation = ori_d
        # post-context
        avg_x = torch.nn.functional.adaptive_avg_pool2d(out, output_size=1)
        avg_x = self.post_context(avg_x)
        avg_x = avg_x.expand_as(out)
        out = out + avg_x
        return self.act(self.bn(out))
#------------------------------------------SAConv2d end------------------------------------

#------------------------------------AAAI 2025 PConv_v1 start---------------------------------
# 论文题目：Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared Small Target Detection
# 风车形状的卷积和基于尺度的动态损失用于红外小目标检测
# 论文链接：https://arxiv.org/pdf/2412.16986
class Conv_v1(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class PConv_v1(nn.Module):  
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''
    
    def __init__(self, c1, c2, k, s):
        super().__init__()

        # self.k = k
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv_v1(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv_v1(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv_v1(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))
#------------------------------------AAAI 2025 PConv_v1 start---------------------------------

#-----------------------------------arXiv 2025 SS-Conv-SSM start------------------------------
# 论文题目：MedMamba: Vision Mamba for Medical Image Classification
# MedMamba: 用于医学图像分类的视觉Mamba 需要安装库mamba_ssm
# 原文：https://arxiv.org/pdf/2403.03849

# import math
# from functools import partial
# from typing import Callable
# from torch import Tensor
# from einops import rearrange, repeat
# from timm.models.layers import DropPath
# try:
#     from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
# except:
#     pass

# # an alternative for mamba_ssm (in which causal_conv1d is needed)
# try:
#     from selective_scan import selective_scan_fn as selective_scan_fn_v1
#     from selective_scan import selective_scan_ref as selective_scan_ref_v1
# except:
#     pass

# class SS2D(nn.Module):
#     def __init__(
#         self,
#         d_model,
#         d_state=16,
#         # d_state="auto", # 20240109
#         d_conv=3,
#         expand=2,
#         dt_rank="auto",
#         dt_min=0.001,
#         dt_max=0.1,
#         dt_init="random",
#         dt_scale=1.0,
#         dt_init_floor=1e-4,
#         dropout=0.,
#         conv_bias=True,
#         bias=False,
#         device=None,
#         dtype=None,
#         **kwargs,
#     ):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.d_model = d_model
#         self.d_state = d_state
#         # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
#         self.d_conv = d_conv
#         self.expand = expand
#         self.d_inner = int(self.expand * self.d_model)
#         self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

#         self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
#         self.conv2d = nn.Conv2d(
#             in_channels=self.d_inner,
#             out_channels=self.d_inner,
#             groups=self.d_inner,
#             bias=conv_bias,
#             kernel_size=d_conv,
#             padding=(d_conv - 1) // 2,
#             **factory_kwargs,
#         )
#         self.act = nn.SiLU()

#         self.x_proj = (
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
#         )
#         self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
#         del self.x_proj

#         self.dt_projs = (
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
#         )
#         self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
#         self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
#         del self.dt_projs
        
#         self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
#         self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

#         # self.selective_scan = selective_scan_fn
#         self.forward_core = self.forward_corev0

#         self.out_norm = nn.LayerNorm(self.d_inner)
#         self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
#         self.dropout = nn.Dropout(dropout) if dropout > 0. else None

#     @staticmethod
#     def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
#         dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

#         # Initialize special dt projection to preserve variance at initialization
#         dt_init_std = dt_rank**-0.5 * dt_scale
#         if dt_init == "constant":
#             nn.init.constant_(dt_proj.weight, dt_init_std)
#         elif dt_init == "random":
#             nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
#         else:
#             raise NotImplementedError

#         # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
#         dt = torch.exp(
#             torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
#             + math.log(dt_min)
#         ).clamp(min=dt_init_floor)
#         # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
#         inv_dt = dt + torch.log(-torch.expm1(-dt))
#         with torch.no_grad():
#             dt_proj.bias.copy_(inv_dt)
#         # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
#         dt_proj.bias._no_reinit = True
        
#         return dt_proj

#     @staticmethod
#     def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
#         # S4D real initialization
#         A = repeat(
#             torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
#             "n -> d n",
#             d=d_inner,
#         ).contiguous()
#         A_log = torch.log(A)  # Keep A_log in fp32
#         if copies > 1:
#             A_log = repeat(A_log, "d n -> r d n", r=copies)
#             if merge:
#                 A_log = A_log.flatten(0, 1)
#         A_log = nn.Parameter(A_log)
#         A_log._no_weight_decay = True
#         return A_log

#     @staticmethod
#     def D_init(d_inner, copies=1, device=None, merge=True):
#         # D "skip" parameter
#         D = torch.ones(d_inner, device=device)
#         if copies > 1:
#             D = repeat(D, "n1 -> r n1", r=copies)
#             if merge:
#                 D = D.flatten(0, 1)
#         D = nn.Parameter(D)  # Keep in fp32
#         D._no_weight_decay = True
#         return D

#     def forward_corev0(self, x: torch.Tensor):
#         self.selective_scan = selective_scan_fn
        
#         B, C, H, W = x.shape
#         L = H * W
#         K = 4

#         x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
#         xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

#         x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
#         # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
#         dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
#         dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
#         # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

#         xs = xs.float().view(B, -1, L) # (b, k * d, l)
#         dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
#         Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
#         Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
#         Ds = self.Ds.float().view(-1) # (k * d)
#         As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
#         dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

#         out_y = self.selective_scan(
#             xs, dts, 
#             As, Bs, Cs, Ds, z=None,
#             delta_bias=dt_projs_bias,
#             delta_softplus=True,
#             return_last_state=False,
#         ).view(B, K, -1, L)
#         assert out_y.dtype == torch.float

#         inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
#         wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
#         invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

#         return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

#     # an alternative to forward_corev1
#     def forward_corev1(self, x: torch.Tensor):
#         self.selective_scan = selective_scan_fn_v1

#         B, C, H, W = x.shape
#         L = H * W
#         K = 4

#         x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
#         xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

#         x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
#         # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
#         dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
#         dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
#         # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

#         xs = xs.float().view(B, -1, L) # (b, k * d, l)
#         dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
#         Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
#         Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
#         Ds = self.Ds.float().view(-1) # (k * d)
#         As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
#         dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

#         out_y = self.selective_scan(
#             xs, dts, 
#             As, Bs, Cs, Ds,
#             delta_bias=dt_projs_bias,
#             delta_softplus=True,
#         ).view(B, K, -1, L)
#         assert out_y.dtype == torch.float

#         inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
#         wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
#         invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

#         return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

#     def forward(self, x: torch.Tensor, **kwargs):
#         B, H, W, C = x.shape

#         xz = self.in_proj(x)
#         x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

#         x = x.permute(0, 3, 1, 2).contiguous()
#         x = self.act(self.conv2d(x)) # (b, d, h, w)
#         y1, y2, y3, y4 = self.forward_core(x)
#         assert y1.dtype == torch.float32
#         y = y1 + y2 + y3 + y4
#         y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
#         y = self.out_norm(y)
#         y = y * F.silu(z)
#         out = self.out_proj(y)
#         if self.dropout is not None:
#             out = self.dropout(out)
#         return out


# def channel_shuffle(x: Tensor, groups: int) -> Tensor:

#     batch_size, height, width, num_channels = x.size()
#     channels_per_group = num_channels // groups

#     # reshape
#     # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
#     x = x.view(batch_size, height, width, groups, channels_per_group)

#     x = torch.transpose(x, 3, 4).contiguous()

#     # flatten
#     x = x.view(batch_size, height, width, -1)

#     return x

# class SS_Conv_SSM(nn.Module):
#     def __init__(
#         self,
#         hidden_dim: int = 0,
#         drop_path: float = 0,
#         norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#         attn_drop_rate: float = 0,
#         d_state: int = 16,
#         **kwargs,
#     ):
#         super().__init__()
#         self.ln_1 = norm_layer(hidden_dim//2)
#         self.self_attention = SS2D(d_model=hidden_dim//2, dropout=attn_drop_rate, d_state=d_state, **kwargs)
#         self.drop_path = DropPath(drop_path)

#         self.conv33conv33conv11 = nn.Sequential(
#             nn.BatchNorm2d(hidden_dim // 2),
#             nn.Conv2d(in_channels=hidden_dim//2,out_channels=hidden_dim//2,kernel_size=3,stride=1,padding=1),
#             nn.BatchNorm2d(hidden_dim//2),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(hidden_dim // 2),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=1, stride=1),
#             nn.ReLU()
#         )
#         # self.finalconv11 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1)
#     def forward(self, input: torch.Tensor):
#         input_left, input_right = input.chunk(2,dim=-1)
#         x = self.drop_path(self.self_attention(self.ln_1(input_right)))
#         input_left = input_left.permute(0,3,1,2).contiguous()
#         input_left = self.conv33conv33conv11(input_left)
#         input_left = input_left.permute(0,2,3,1).contiguous()
#         output = torch.cat((input_left,x),dim=-1)
#         output = channel_shuffle(output,groups=2)
#         return output+input
# ------------------------------------arXiv 2025 SS-Conv-SSM end--------------------------------------

# ------------------------------------arXiv 2023 RefConv start----------------------------------------
# 论文题目：RefConv: Re-parameterized Refocusing Convolution for Powerful ConvNets
# 中文题目:  RefConv：用于强大卷积网络的重新参数化重聚焦卷积
# 论文链接：https://arxiv.org/pdf/2310.10563
# 官方github：https://github.com/Aiolus-X/RefConv

class RepConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, groups=1,
                 map_k=3):
        super(RepConv, self).__init__()
        assert map_k <= kernel_size
        self.origin_kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        self.register_buffer('weight', torch.zeros(*self.origin_kernel_shape))
        G = in_channels * out_channels // (groups ** 2)
        self.num_2d_kernels = out_channels * in_channels // groups
        self.kernel_size = kernel_size
        self.convmap = nn.Conv2d(in_channels=self.num_2d_kernels,
                                 out_channels=self.num_2d_kernels, kernel_size=map_k, stride=1, padding=map_k // 2,
                                 groups=G, bias=False)
        #nn.init.zeros_(self.convmap.weight)
        self.bias = None#nn.Parameter(torch.zeros(out_channels), requires_grad=True)     # must have a bias for identical initialization
        self.stride = stride
        self.groups = groups
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

    def forward(self, inputs):
        origin_weight = self.weight.view(1, self.num_2d_kernels, self.kernel_size, self.kernel_size)
        kernel = self.weight + self.convmap(origin_weight).view(*self.origin_kernel_shape)
        return F.conv2d(inputs, kernel, stride=self.stride, padding=self.padding, dilation=1, groups=self.groups, bias=self.bias)


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        RepConv(inp, oup, kernel_size=3, stride=stride, padding=None, groups=1, map_k=3),
        #conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.
#------------------------------------arXiv 2023 RefConv end----------------------------------------

# ------------------------------------arXiv 2020 CondConv start----------------------------------------
import functools
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

# 论文题目：SMFANet: A Lightweight Self-Modulation Feature Aggregation Network for Efficient Image Super-Resolution
# 中文题目:  CondConv:用于高效推理的条件参数化卷积
# 论文链接：https://arxiv.org/pdf/1904.04971
# 官方github：https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/condconv


class _routing(nn.Module):
    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)  # 定义Dropout层
        self.fc = nn.Linear(in_channels, num_experts)  # 定义全连接层

    def forward(self, x):
        x = torch.flatten(x)  # 将输入展平
        x = self.dropout(x)  # 应用Dropout
        x = self.fc(x)  # 通过全连接层
        return torch.sigmoid(x)  # 使用Sigmoid激活函数

class CondConv2D(_ConvNd):
    r"""
    参数：
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int或tuple): 卷积核大小
        stride (int或tuple, 可选): 卷积步幅，默认值为1
        padding (int或tuple, 可选): 输入两侧的零填充，默认值为0
        padding_mode (str, 可选): 填充模式，如'zeros'、'reflect'等，默认值为'zeros'
        dilation (int或tuple, 可选): 卷积核元素间距，默认值为1
        groups (int, 可选): 输入输出通道分组数量，默认值为1
        bias (bool, 可选): 是否添加偏置项，默认值为True
        num_experts (int): 每层的专家数量
        dropout_rate (float): Dropout的概率

    输入输出形状：
        输入：形状为(N, C_in, H_in, W_in)
        输出：形状为(N, C_out, H_out, W_out)

    属性：
        weight (Tensor): 学习的卷积核权重
        bias (Tensor): 可选的偏置项
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', num_experts=3, dropout_rate=0.2):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(CondConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))  # 自适应平均池化
        self._routing_fn = _routing(in_channels, num_experts, dropout_rate)  # 路由函数

        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))  # 初始化卷积核权重

        self.reset_parameters()  # 重置参数

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':  # 如果不是零填充，执行自定义填充模式
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride, _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, inputs):
        b, _, _, _ = inputs.size()  # 获取输入的批大小
        res = []
        for input in inputs:
            input = input.unsqueeze(0)  # 增加一个维度
            pooled_inputs = self._avg_pooling(input)  # 自适应平均池化
            routing_weights = self._routing_fn(pooled_inputs)  # 生成路由权重
            kernels = torch.sum(routing_weights[:, None, None, None, None] * self.weight, 0)  # 加权卷积核
            out = self._conv_forward(input, kernels)  # 执行卷积操作
            # print(f"CondConv2D output size: {out.size()}")  # 添加打印语句
            res.append(out)  # 存储结果
        return torch.cat(res, dim=0)  # 拼接结果返回
#------------------------------------arXiv 2020 CondConv end----------------------------------------

#--------------------------------------CVPR 2024 InceptionDWConv2d 2024 start-----------------------
# 论文题目：InceptionNeXt: When Inception Meets ConvNeXt
# 论文链接：https://arxiv.org/pdf/2303.16900
# 官方github：https://github.com/sail-sg/inceptionnext

class InceptionDWConv2d(nn.Module):
    """ 
    Inception深度可分离卷积模块
    这个模块使用多个分支来对输入进行不同类型的卷积操作，类似于Inception模块的结构。
    每个分支使用深度卷积（depthwise convolution），以便减少计算量。
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        # 根据branch_ratio计算每个卷积分支的输出通道数
        gc = int(in_channels * branch_ratio)  # branch的通道数
        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        # 计算每个分支的输入输出通道数：idc (原始通道数 - 3个分支的通道数)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        """
        前向传播：对输入x进行拆分并通过各分支卷积处理，最后合并结果
        """
        # 根据计算的split_indexes拆分输入x
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        
        # 将原始通道x_id与经过3个卷积分支处理后的x_hw、x_w、x_h合并
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,  # 在通道维度上拼接
        )
#--------------------------------------CVPR 2024 InceptionDWConv2d 2024 end-----------------------

#--------------------------------------ICLR 2022 ODConv_3rdstart-------------------------------
import math
class ODConv2d_3rd(nn.Conv2d):
 
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 K=4, r=1 / 16, save_parameters=False,
                 padding_mode='zeros', device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.K = K
        self.r = r
        self.save_parameters = save_parameters
 
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)
 
        del self.weight
        self.weight = nn.Parameter(torch.empty((
            K,
            out_channels,
            in_channels // groups,
            *self.kernel_size,
        ), **factory_kwargs))
 
        if bias:
            del self.bias
            self.bias = nn.Parameter(torch.empty(K, out_channels, **factory_kwargs))
 
        hidden_dim = max(int(in_channels * r), 16)  #设置下限为16
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.reduction = nn.Linear(in_channels, hidden_dim)
        self.fc = nn.Conv2d(in_channels, hidden_dim, 1, bias = False)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        # self.act = nn.SiLU(inplace=True)
 
        self.fc_f = nn.Linear(hidden_dim, out_channels)
        if not save_parameters or self.kernel_size[0] * self.kernel_size[1] > 1:
            self.fc_s = nn.Linear(hidden_dim, self.kernel_size[0] * self.kernel_size[1])
        if not save_parameters or in_channels // groups > 1:
            self.fc_c = nn.Linear(hidden_dim, in_channels // groups)
        if not save_parameters or K > 1:
            self.fc_w = nn.Linear(hidden_dim, K)
 
        self.reset_parameters()
 
    def reset_parameters(self) -> None:
        fan_out = self.kernel_size[0] * self.kernel_size[1] * self.out_channels // self.groups
        for i in range(self.K):
            self.weight.data[i].normal_(0, math.sqrt(2.0 / fan_out))
        if self.bias is not None:
            self.bias.data.zero_()
 
    def extra_repr(self):
        return super().extra_repr() + f', K={self.K}, r={self.r:.4}'
 
    def get_weight_bias(self, context):
        B, C, H, W = context.shape
 
        if C != self.in_channels:
            raise ValueError(
                f"Expected context{[B, C, H, W]} to have {self.in_channels} channels, but got {C} channels instead")
 
        # x = self.gap(context).squeeze(-1).squeeze(-1)  # B, c_in
        # x = self.reduction(x)  # B, hidden_dim
        x = self.gap(context)
        x = self.fc(x)
        if x.size(0)>1:
            x = self.bn(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.act(x)
 
        attn_f = self.fc_f(x).sigmoid()  # B, c_out
        attn = attn_f.view(B, 1, -1, 1, 1, 1)  # B, 1, c_out, 1, 1, 1
        if hasattr(self, 'fc_s'):
            attn_s = self.fc_s(x).sigmoid()  # B, k * k
            attn = attn * attn_s.view(B, 1, 1, 1, *self.kernel_size)  # B, 1, c_out, 1, k, k
        if hasattr(self, 'fc_c'):
            attn_c = self.fc_c(x).sigmoid()  # B, c_in // groups
            attn = attn * attn_c.view(B, 1, 1, -1, 1, 1)  # B, 1, c_out, c_in // groups, k, k
        if hasattr(self, 'fc_w'):
            attn_w = self.fc_w(x).softmax(-1)  # B, n
            attn = attn * attn_w.view(B, -1, 1, 1, 1, 1)  # B, n, c_out, c_in // groups, k, k
 
        weight = (attn * self.weight).sum(1)  # B, c_out, c_in // groups, k, k
        weight = weight.view(-1, self.in_channels // self.groups, *self.kernel_size)  # B * c_out, c_in // groups, k, k
 
        bias = None
        if self.bias is not None:
            if hasattr(self, 'fc_w'):
                bias = attn_w @ self.bias
            else:
                bias = self.bias.tile(B, 1)
            bias = bias.view(-1)  # B * c_out
 
        return weight, bias
 
    def forward(self, input, context=None):
        B, C, H, W = input.shape
 
        if C != self.in_channels:
            raise ValueError(
                f"Expected input{[B, C, H, W]} to have {self.in_channels} channels, but got {C} channels instead")
 
        weight, bias = self.get_weight_bias(context or input)
 
        output = nn.functional.conv2d(
            input.view(1, B * C, H, W), weight, bias,
            self.stride, self.padding, self.dilation, B * self.groups)  # 1, B * c_out, h_out, w_out
        output = output.view(B, self.out_channels, *output.shape[2:])
 
        return output
 
    def debug(self, input, context=None):
        B, C, H, W = input.shape
 
        if C != self.in_channels:
            raise ValueError(
                f"Expected input{[B, C, H, W]} to have {self.in_channels} channels, but got {C} channels instead")
 
        output_size = [
            ((H, W)[i] + 2 * self.padding[i] - self.dilation[i] * (self.kernel_size[i] - 1) - 1) // self.stride[i] + 1
            for i in range(2)
        ]
 
        weight, bias = self.get_weight_bias(context or input)
 
        weight = weight.view(B, self.groups, self.out_channels // self.groups, -1)  # B, groups, c_out // groups, c_in // groups * k * k
 
        unfold = nn.functional.unfold(
            input, self.kernel_size, self.dilation, self.padding, self.stride)  # B, c_in * k * k, H_out * W_out
        unfold = unfold.view(B, self.groups, -1, output_size[0] * output_size[1])  # B, groups, c_in // groups * k * k, H_out * W_out
 
        output = weight @ unfold  # B, groups, c_out // groups, H_out * W_out
        output = output.view(B, self.out_channels, *output_size)  # B, c_out, H_out * W_out
 
        if bias is not None:
            output = output + bias.view(B, self.out_channels, 1, 1)
 
        return output
 
class ODConv_3rd(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, kerNums=1, g=1, p=None, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = ODConv2d_3rd(c1, c2, k, s, autopad(k, p), groups=g, K=kerNums)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
 
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        return self.act(self.conv(x))
#--------------------------------------ICLR 2022 ODConv_3rd start-------------------------------


#--------------------------------------IVC 2024 LDConv start------------------------------------
from einops import rearrange
import math

# 论文题目：LDConv: Linear deformable convoluton for improving convolutioanl neural networks
# 中文题目: LDConv：用于改进卷积神经网络的线性可变形卷积
# 论文链接：https://doi.org/10.1016/j.imavis.2024.105190

class LDConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(LDConv, self).__init__()
        self.num_param = num_param  # 参数个数
        self.stride = stride  # 步长
        # 定义卷积层，包含BN层和SiLU激活函数，用于与原始YOLOv5的卷积进行比较
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
            nn.BatchNorm2d(outc),
            nn.SiLU()
        )
        # 定义偏移量卷积层，用于生成2 * num_param个偏移参数
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)  # 初始化偏移量卷积的权重为0
        # 注册反向传播时的钩子函数，调整学习率
        # self.p_conv.register_full_backward_hook(self._set_lr)

    # 静态方法，用于在反向传播时调整梯度，减小梯度幅度
    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N 表示参数数量
        offset = self.p_conv(x)  # 生成偏移量
        dtype = offset.data.type()
        N = offset.size(1) // 2  # 偏移量的个数
        # 根据偏移量生成坐标
        p = self._get_p(offset, dtype)

        # 将坐标排列为 (b, h, w, 2N) 的格式
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()  # 左上角坐标取整
        q_rb = q_lt + 1  # 右下角坐标

        # 限制坐标范围，使其不超出输入图像的边界
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)  # 左下角坐标
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)  # 右上角坐标

        # 将坐标p限制在图像范围内
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # 双线性插值系数
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # 根据调整后的坐标重新采样特征
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # 双线性插值
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # 调整偏移特征的形状
        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)  # 最终卷积操作

        return out

    # 生成不同大小的初始采样形状
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))  # 基准整形数
        row_number = self.num_param // base_int  # 行数
        mod_number = self.num_param % base_int  # 剩余数量
        # 创建网格坐标
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int),
            indexing='ij'  # 显式指定索引模式
        )
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            # 如果有余数，将剩余部分也加入网格
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number+1),
                torch.arange(0, mod_number)
            )
            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)  # 合并坐标
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)  # 重塑形状
        return p_n

    # 无填充
    def _get_p_0(self, h, w, N, dtype):
        # 生成坐标网格
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride),
            indexing='ij'
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    # 计算实际偏移坐标
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)  # 获取基础采样点
        p_0 = self._get_p_0(h, w, N, dtype)  # 获取网格坐标
        p = p_0 + p_n + offset  # 加上偏移量
        return p

    # 获取在新坐标上的特征值
    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)  # 展平特征

        # 计算索引值
        index = q[..., :N] * padded_w + q[..., N:]  # 计算偏移索引
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        # 获取偏移后的特征
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    
    # Stacking resampled features in the row direction.
    # 在行方向上堆叠重采样的特征
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()  # 获取 x_offset 张量的尺寸参数：批次大小(b)、通道数(c)、高度(h)、宽度(w)和重采样数量(n)
        
        # using Conv3d
        # 使用 3D 卷积操作：将 x_offset 进行维度排列 (b, c, n, h, w)，然后使用 Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c, c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        
        # using 1 × 1 Conv
        # 使用 1 × 1 卷积：将 x_offset 进行维度排列 (b, c, n, h, w)，然后将 x_offset 重塑为 (b, c×num_param, h, w)
        # 然后使用 Conv2d 卷积核大小为1×1的操作完成特征变换
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b, c×num_param, h, w)  
        # finally, Conv2d(c×num_param, c_out, kernel_size =1, stride=1, bias= False)

        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)
        # 使用列卷积：首先进行列方向上的卷积，接着使用 kernel_size=(num_param, 1) 和 stride=(num_param, 1) 的 Conv2d
        
        # Rearrange x_offset dimensions for column stacking.
        # 使用 einops 库的 rearrange 方法，将 x_offset 从 (b, c, h, w, n) 重新排列为 (b, c, h×n, w)
        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        
        return x_offset  # 返回重新排列后的 x_offset

#--------------------------------------IVC 2024 LDConv end------------------------------------

#--------------------------------------CVPR 2023 Partial_conv3 start-------------------------

#论文地址：https://arxiv.org/pdf/2303.03667
#GitHub地址：https://github.com/JierunChen/FasterNet

class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div=2, forward='split_cat'):
        super().__init__()
        # 计算被部分卷积处理的通道数
        self.dim_conv3 = dim // n_div
        # 计算未被处理的通道数
        self.dim_untouched = dim - self.dim_conv3
        # 定义部分卷积层，卷积核大小为3，步幅为1，填充为1
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        # 根据选择的前向传播方式定义前向传播方法
        if forward == 'slicing':
            self.forward = self.forward_slicing  # 用于推断
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat  # 用于训练或推断
        else:
            raise NotImplementedError  # 抛出未实现的异常

    def forward_slicing(self, x):
        # 仅用于推断阶段
        x = x.clone()  # !!! 保持原始输入不变，以便后续的残差连接
        # 处理部分卷积的通道
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x  # 返回处理后的特征图

    def forward_split_cat(self, x):
        # 用于训练/推断阶段
        # 将输入张量拆分为两部分
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        # 处理部分卷积的通道
        x1 = self.partial_conv3(x1)
        # 将处理后的部分与未处理的部分拼接在一起
        x = torch.cat((x1, x2), 1)

        return x  # 返回拼接后的特征图
#--------------------------------------CVPR 2023 Partial_conv3 end-------------------------
from models.common import ELA
# 改1
class Partial_conv3_ELA(nn.Module):
    def __init__(self, dim, n_div=2, forward='split_cat'):
        super().__init__()
        # 原Partial_conv3结构
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        
        # 新增ELA分支
        self.ela = ELA(self.dim_conv3)  # 注意力增强分支
        
        # 自适应融合参数
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.forward = self.forward_split_cat if forward == 'split_cat' else self.forward_slicing

    def forward_split_cat(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        
        # 主路径：常规卷积
        main_feat = self.partial_conv3(x1)
        
        # 辅助路径：ELA增强
        attn_feat = self.ela(x1) 
        
        # 自适应融合
        fused_feat = main_feat + self.gamma * attn_feat
        
        return torch.cat([fused_feat, x2], dim=1)

    def forward_slicing(self, x):
        x = x.clone()
        slice_part = x[:, :self.dim_conv3]
        main_feat = self.partial_conv3(slice_part)
        attn_feat = self.ela(slice_part)
        x[:, :self.dim_conv3] = main_feat + self.gamma * attn_feat
        return x

# 改2
class Pconv_ELA(nn.Module):
    def __init__(self, dim, n_div=2, forward='split_cat'):
        super().__init__()
        # 计算被部分卷积处理的通道数
        self.dim_conv3 = dim // n_div
        # 计算未被处理的通道数
        self.dim_untouched = dim - self.dim_conv3
        # 定义部分卷积层，卷积核大小为3，步幅为1，填充为1
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        # 初始化 ELA 模块
        self.ela = ELA(self.dim_untouched)

        # 根据选择的前向传播方式定义前向传播方法
        if forward == 'slicing':
            self.forward = self.forward_slicing  # 用于推断
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat  # 用于训练或推断
        else:
            raise NotImplementedError  # 抛出未实现的异常

    def forward_slicing(self, x):
        # 仅用于推断阶段
        x = x.clone()  # !!! 保持原始输入不变，以便后续的残差连接
        # 处理部分卷积的通道
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x  # 返回处理后的特征图

    def forward_split_cat(self, x):
        # 用于训练/推断阶段
        # 将输入张量拆分为两部分
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        # 处理部分卷积的通道
        x1 = self.partial_conv3(x1)
        # 对 x2 应用 ELA 模块
        x2 = self.ela(x2)
        # 将处理后的部分与经过 ELA 处理的部分拼接在一起
        x = torch.cat((x1, x2), 1)

        return x  # 返回拼接后的特征图


#--------------------------------------WTConv2d ECCV 2024 start----------------------------
import pywt
import pywt.data
from functools import partial

# 论文地址 https://arxiv.org/pdf/2407.05848


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


# Wavelet Transform Conv(WTConv2d)
class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class DepthwiseSeparableConvWithWTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DepthwiseSeparableConvWithWTConv2d, self).__init__()

        # 深度卷积：使用 WTConv2d 替换 3x3 卷积
        self.depthwise = WTConv2d(in_channels, in_channels, kernel_size=kernel_size)

        # 逐点卷积：使用 1x1 卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
#--------------------------------------WTConv2d ECCV 2024 end----------------------------

######################################## C3 C2f DCNV2 start ########################################
class BottleneckDCNV2(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3DCNV2(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(BottleneckDCNV2(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, groups=1, dilation=1, act=True, deformable_groups=1):
        super(DCNv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        padding = autopad(kernel_size, padding, dilation)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() 
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

class Bottleneck_DCNV2(BottleneckDCNV2):
    """Standard bottleneck with DCNV2."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DCNv2(c_, c2, k[1], 1)

class C3_DCNv2(DCNv2):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DCNV2(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))

# class C2f_DCNv2(C2f):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.m = nn.ModuleList(Bottleneck_DCNV2(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

######################################## C3 C2f DCNV2 end ########################################

###################### GFPN  strat###############################
 
def conv_bnGFPN(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module(
        'conv',
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  groups=groups,
                  bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result
 
class RepConv_GFPN(nn.Module):
    '''RepConv is a basic rep-style block, including training and deploy status
    Code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 deploy=False,
                 act='relu',
                 norm=None):
        super(RepConv_GFPN, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
 
        assert kernel_size == 3
        assert padding == 1
 
        padding_11 = padding - kernel_size // 2
 
        if isinstance(act, str):
            self.nonlinearity = get_activation(act)
        else:
            self.nonlinearity = act
 
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=True,
                                         padding_mode=padding_mode)
 
        else:
            self.rbr_identity = None
            self.rbr_dense = conv_bnGFPN(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     groups=groups)
            self.rbr_1x1 = conv_bnGFPN(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=stride,
                                   padding=padding_11,
                                   groups=groups)
 
    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))
 
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
 
        return self.nonlinearity(
            self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
 
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(
            kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
 
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
 
    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
 
    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True
 
class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace
 
    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)
 
def get_activation(name='silu', inplace=True):
    if name is None:
        return nn.Identity()
 
    if isinstance(name, str):
        if name == 'silu':
            module = nn.SiLU(inplace=inplace)
        elif name == 'relu':
            module = nn.ReLU(inplace=inplace)
        elif name == 'lrelu':
            module = nn.LeakyReLU(0.1, inplace=inplace)
        elif name == 'swish':
            module = Swish(inplace=inplace)
        elif name == 'hardsigmoid':
            module = nn.Hardsigmoid(inplace=inplace)
        elif name == 'identity':
            module = nn.Identity()
        else:
            raise AttributeError('Unsupported act type: {}'.format(name))
        return module
 
    elif isinstance(name, nn.Module):
        return name
 
    else:
        raise AttributeError('Unsupported act type: {}'.format(name))
 
def get_norm(name, out_channels, inplace=True):
    if name == 'bn':
        module = nn.BatchNorm2d(out_channels)
    else:
        raise NotImplementedError
    return module
 
class ConvBNAct(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""
    def __init__(
        self,
        in_channels,
        out_channels,
        ksize,
        stride=1,
        groups=1,
        bias=False,
        act='silu',
        norm='bn',
        reparam=False,
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        if norm is not None:
            self.bn = get_norm(norm, out_channels, inplace=True)
        if act is not None:
            self.act = get_activation(act, inplace=True)
        self.with_norm = norm is not None
        self.with_act = act is not None
 
    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:
            x = self.bn(x)
        if self.with_act:
            x = self.act(x)
        return x
 
    def fuseforward(self, x):
        return self.act(self.conv(x))
 
class BasicBlock_3x3_Reverse(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_hidden_ratio,
                 ch_out,
                 act='relu',
                 shortcut=True):
        super(BasicBlock_3x3_Reverse, self).__init__()
        assert ch_in == ch_out
        ch_hidden = int(ch_in * ch_hidden_ratio)
        self.conv1 = ConvBNAct(ch_hidden, ch_out, 3, stride=1, act=act)
        self.conv2 = RepConv_GFPN(ch_in, ch_hidden, 3, stride=1, act=act)
        self.shortcut = shortcut
 
    def forward(self, x):
        y = self.conv2(x)
        y = self.conv1(y)
        if self.shortcut:
            return x + y
        else:
            return y
 
class SPP(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        k,
        pool_size,
        act='swish',
    ):
        super(SPP, self).__init__()
        self.pool = []
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(kernel_size=size,
                                stride=1,
                                padding=size // 2,
                                ceil_mode=False)
            self.add_module('pool{}'.format(i), pool)
            self.pool.append(pool)
        self.conv = ConvBNAct(ch_in, ch_out, k, act=act)
 
    def forward(self, x):
        outs = [x]
 
        for pool in self.pool:
            outs.append(pool(x))
        y = torch.cat(outs, axis=1)
 
        y = self.conv(y)
        return y
 
class CSPStage(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 n=1,
                 block_fn='BasicBlock_3x3_Reverse',
                 ch_hidden_ratio=1.0,
                 act='silu',
                 spp=False):
        super(CSPStage, self).__init__()
 
        split_ratio = 2
        ch_first = int(ch_out // split_ratio)
        ch_mid = int(ch_out - ch_first)
        self.conv1 = ConvBNAct(ch_in, ch_first, 1, act=act)
        self.conv2 = ConvBNAct(ch_in, ch_mid, 1, act=act)
        self.convs = nn.Sequential()
 
        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == 'BasicBlock_3x3_Reverse':
                self.convs.add_module(
                    str(i),
                    BasicBlock_3x3_Reverse(next_ch_in,
                                           ch_hidden_ratio,
                                           ch_mid,
                                           act=act,
                                           shortcut=True))
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_module(
                    'spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act))
            next_ch_in = ch_mid
        self.conv3 = ConvBNAct(ch_mid * n + ch_first, ch_out, 1, act=act)
 
    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
 
        mid_out = [y1]
        for conv in self.convs:
            y2 = conv(y2)
            mid_out.append(y2)
        y = torch.cat(mid_out, axis=1)
        y = self.conv3(y)
        return y
 
 
###################### GFPN  END   ###############################

######################################## BIFPN begin ########################################
class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, Conv.default_act)
        self.cv2 = Conv(c_, c_, 5, 1, p, c_, d, Conv.default_act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])

        b, n, h, w = x2.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)

class SDI(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # self.convs = nn.ModuleList([nn.Conv2d(channel, channels[0], kernel_size=3, stride=1, padding=1) for channel in channels])
        self.convs = nn.ModuleList([GSConv(channel, channels[0]) for channel in channels])

    def forward(self, xs):
        ans = torch.ones_like(xs[0])
        target_size = xs[0].shape[2:]
        for i, x in enumerate(xs):
            if x.shape[-1] > target_size[-1]:
                x = F.adaptive_avg_pool2d(x, (target_size[0], target_size[1]))
            elif x.shape[-1] < target_size[-1]:
                x = F.interpolate(x, size=(target_size[0], target_size[1]),
                                      mode='bilinear', align_corners=True)
            ans = ans * self.convs[i](x)
        return ans

class Fusion(nn.Module):
    def __init__(self, inc_list, fusion='bifpn') -> None:
        super().__init__()
        
        assert fusion in ['weight', 'adaptive', 'concat', 'bifpn', 'SDI']
        self.fusion = fusion
        
        if self.fusion == 'bifpn':
            self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
            self.relu = nn.ReLU()
            self.epsilon = 1e-4
        elif self.fusion == 'SDI':
            self.SDI = SDI(inc_list)
        else:
            self.fusion_conv = nn.ModuleList([Conv(inc, inc, 1) for inc in inc_list])

            if self.fusion == 'adaptive':
                self.fusion_adaptive = Conv(sum(inc_list), len(inc_list), 1)
        
    
    def forward(self, x):
        if self.fusion in ['weight', 'adaptive']:
            for i in range(len(x)):
                x[i] = self.fusion_conv[i](x[i])
        if self.fusion == 'weight':
            return torch.sum(torch.stack(x, dim=0), dim=0)
        elif self.fusion == 'adaptive':
            fusion = torch.softmax(self.fusion_adaptive(torch.cat(x, dim=1)), dim=1)
            x_weight = torch.split(fusion, [1] * len(x), dim=1)
            return torch.sum(torch.stack([x_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'concat':
            return torch.cat(x, dim=1)
        elif self.fusion == 'bifpn':
            fusion_weight = self.relu(self.fusion_weight.clone())
            fusion_weight = fusion_weight / (torch.sum(fusion_weight, dim=0) + self.epsilon)
            return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'SDI':
            return self.SDI(x)

######################################## BIFPN end ########################################

# ######################################## HS-FPN start ########################################

# class ChannelAttention_HSFPN(nn.Module):
#     def __init__(self, in_planes, ratio = 4, flag=True):
#         super(ChannelAttention_HSFPN, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.conv1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#         self.flag = flag
#         self.sigmoid = nn.Sigmoid()

#         nn.init.xavier_uniform_(self.conv1.weight)
#         nn.init.xavier_uniform_(self.conv2.weight)

#     def forward(self, x):
#         avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
#         max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out) * x if self.flag else self.sigmoid(out)

# class ELA_HSFPN(nn.Module):
#     def __init__(self, in_planes, flag=True):
#         super(ELA_HSFPN, self).__init__()
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#         self.conv1x1 = nn.Sequential(
#             nn.Conv1d(in_planes, in_planes, 7, padding=3),
#             nn.GroupNorm(16, in_planes),
#             nn.Sigmoid()
#         )
#         self.flag = flag
    
#     def forward(self, x):
#         b, c, h, w = x.size()
#         x_h = self.conv1x1(self.pool_h(x).reshape((b, c, h))).reshape((b, c, h, 1))
#         x_w = self.conv1x1(self.pool_w(x).reshape((b, c, w))).reshape((b, c, 1, w))
#         return x * x_h * x_w if self.flag else x_h * x_w

# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)

#     def forward(self, x):
#         return self.relu(x + 3) / 6


# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)

#     def forward(self, x):
#         return x * self.sigmoid(x)

# class CA_HSFPN(nn.Module):
#     def __init__(self, inp, reduction=8, flag=True):
#         super(CA_HSFPN, self).__init__()
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#         mip = max(8, inp // reduction)
#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = h_swish()
#         self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

#         self.flag = flag
    
#     def forward(self, x):
#         n, c, h, w = x.size()
#         x_h = self.pool_h(x)
#         x_w = self.pool_w(x).permute(0, 1, 3, 2)

#         y = torch.cat([x_h, x_w], dim=2)
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.act(y)

#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)

#         a_h = self.conv_h(x_h).sigmoid()
#         a_w = self.conv_w(x_w).sigmoid()

#         out = a_w * a_h
#         return x * out if self.flag else out

# class CAA_HSFPN(nn.Module):
#     def __init__(self, ch, flag=True, h_kernel_size = 11, v_kernel_size = 11) -> None:
#         super(CAA_HSFPN, self).__init__()
        
#         self.avg_pool = nn.AvgPool2d(7, 1, 3)
#         self.conv1 = Conv(ch, ch)
#         self.h_conv = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, h_kernel_size // 2), 1, ch)
#         self.v_conv = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), 1, ch)
#         self.conv2 = Conv(ch, ch)
#         self.act = nn.Sigmoid()
        
#         self.flag = flag
    
#     def forward(self, x):
#         out = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
#         return out * x if self.flag else out

# class Multiply(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
    
#     def forward(self, x):
#         return x[0] * x[1]

# class Add_HSFPN(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return torch.sum(torch.stack(x, dim=0), dim=0)
    

# ######################################## HS-FPN end ########################################

#--------------------------------ICLR 2024 MogaNet start------------------------------------------------
# 论文：MogaNet: Multi-order Gated Aggregation Network (ICLR 2024)
# 论文地址：https://arxiv.org/pdf/2211.03295
# Github地址：https://github.com/Westlake-AI/MogaNet
# Spatial Block with Multi-order Gated Aggregation
def build_act_layer(act_type):
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()

class ElementScale(nn.Module):
    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

class MultiOrderDWConv(nn.Module):

    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3,],
                 channel_split=[1, 3, 4,],
                ):
        super(MultiOrderDWConv, self).__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # basic DW conv
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims,
            stride=1, dilation=dw_dilation[0],
        )
        # DW conv 1
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1, dilation=dw_dilation[1],
        )
        # DW conv 2
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1, dilation=dw_dilation[2],
        )
        # a channel convolution
        self.PW_conv = nn.Conv2d(  # point-wise convolution
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0: self.embed_dims_0+self.embed_dims_1, ...])
        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims-self.embed_dims_2:, ...])
        x = torch.cat([
            x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x

class MultiOrderGatedAggregation(nn.Module):

    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False,
                ):
        super(MultiOrderGatedAggregation, self).__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )
        self.proj_2 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # activation for gating and value
        self.act_value = build_act_layer(attn_act_type)
        self.act_gate = build_act_layer(attn_act_type)

        # decompose
        self.sigma = ElementScale(
            embed_dims, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        # x_d: [B, C, H, W] -> [B, C, 1, 1]
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x

    def forward_gating(self, g, v):
        with torch.autocast(device_type='cuda', enabled=False):
            g = g.to(torch.float32)
            v = v.to(torch.float32)
            return self.proj_2(self.act_gate(g) * self.act_gate(v))

    def forward(self, x):
        shortcut = x.clone()
        # proj 1x1
        x = self.feat_decompose(x)
        # gating and value branch
        g = self.gate(x)
        v = self.value(x)
        # aggregation
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(self.act_gate(g), self.act_gate(v))
        x = x + shortcut
        return x
    
class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        act_type (str): The type of activation. Defaults to 'GELU'.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 kernel_size=3,
                 act_type='GELU',
                 ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = int(embed_dims * 4)

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = build_act_layer(act_type)

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
#--------------------------------ICLR 2024 MogaNet end-------------------------------------------