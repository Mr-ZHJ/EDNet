import math
import torch
import torch.nn as nn
from models.common import Conv, Bottleneck, C3, autopad
from models.Slim import GSConv

# ---------------------------GcNet（C3GC) start-------------------------------------------
class ContextBlock2d(nn.Module):
    """ContextBlock2d
    Parameters
    ----------
    inplanes : int
        Number of in_channels.
    pool : string
        spatial att or global pooling (default:'att').
    fusions : list
    Reference:
        Yue Cao, et al. "GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond."
    """

    def __init__(self, inplanes, pool='att', fusions=None):
        super(ContextBlock2d, self).__init__()
        if fusions is None:
            fusions = ['channel_add', 'channel_mul']
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = inplanes // 4
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                                                  nn.LayerNorm([self.planes, 1, 1]), nn.SiLU(inplace=True),
                                                  nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                                                  nn.LayerNorm([self.planes, 1, 1]), nn.SiLU(inplace=True),
                                                  nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


class C3GC(nn.Module):
    # C3 Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3GC, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.gc = ContextBlock2d(c1)
        self.cv1 = Conv(c1, c_)
        self.cv2 = Conv(c1, c_)
        self.cv3 = Conv(2 * c_, c2)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        out = torch.cat((self.m(self.cv1(x)), self.cv2(self.gc(x))), dim=1)
        out = self.cv3(out)
        return out

# ---------------------------GcNet（C3GC) end-------------------------------------------

######################################## ContextGuidedBlock start ########################################

def myautopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class myConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, myautopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class myBottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = myConv(c1, c_, k[0], 1)
        self.cv2 = myConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class myC3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = myConv(c1, c_, 1, 1)
        self.cv2 = myConv(c1, c_, 1, 1)
        self.cv3 = myConv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(myBottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


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
        self.conv1x1 = myConv(nIn, n, 1, 1)  #1x1 Conv is employed to reduce the computation
        self.F_loc = nn.Conv2d(n, n, 3, padding=1, groups=n)
        self.F_sur = nn.Conv2d(n, n, 3, padding=myautopad(3, None, dilation_rate), dilation=dilation_rate, groups=n) # surrounding context
        self.bn_act = nn.Sequential(
            nn.BatchNorm2d(nOut),
            myConv.default_act
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
    def __init__(self, nIn, dilation_rate=2, reduction=16):
        """
        args:
           nIn: the channel of input feature map
           nOut: the channel of output feature map, and nOut=2*nIn
        """
        super().__init__()
        nOut = 2 * nIn
        self.conv1x1 = myConv(nIn, nOut, 3, s=2)  #  size/2, channel: nIn--->nOut
        
        self.F_loc = nn.Conv2d(nOut, nOut, 3, padding=1, groups=nOut)
        self.F_sur = nn.Conv2d(nOut, nOut, 3, padding=myautopad(3, None, dilation_rate), dilation=dilation_rate, groups=nOut) 
        
        self.bn = nn.BatchNorm2d(2 * nOut, eps=1e-3)
        self.act = myConv.default_act
        self.reduce = myConv(2 * nOut, nOut,1,1)  #reduce dimension: 2*nOut--->nOut
        
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

        return output

class C3_ContextGuided(myC3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(ContextGuidedBlock(c_, c_) for _ in range(n)))


######################################## ContextGuidedBlock end ########################################

######################################## ScConv begin ########################################
import torch.nn.functional as F

# CVPR2023 https://openaccess.thecvf.com/content/CVPR2023/papers/Li_SCConv_Spatial_and_Channel_Reconstruction_Convolution_for_Feature_Redundancy_CVPR_2023_paper.pdf
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num:int, 
                 group_num:int = 16, 
                 eps:float = 1e-10
                 ):
        super(GroupBatchnorm2d,self).__init__()
        assert c_num    >= group_num
        self.group_num  = group_num
        self.gamma      = nn.Parameter(torch.randn(c_num, 1, 1))
        self.beta       = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps        = eps

    def forward(self, x):
        N, C, H, W  = x.size()
        x           = x.view(   N, self.group_num, -1   )
        mean        = x.mean(   dim = 2, keepdim = True )
        std         = x.std (   dim = 2, keepdim = True )
        x           = (x - mean) / (std+self.eps)
        x           = x.view(N, C, H, W)
        return x * self.gamma + self.beta

class SRU(nn.Module):
    def __init__(self,
                 oup_channels:int, 
                 group_num:int = 16,
                 gate_treshold:float = 0.5 
                 ):
        super().__init__()
        
        self.gn             = GroupBatchnorm2d( oup_channels, group_num = group_num )
        self.gate_treshold  = gate_treshold
        self.sigomid        = nn.Sigmoid()

    def forward(self,x):
        gn_x        = self.gn(x)
        w_gamma     = self.gn.gamma/sum(self.gn.gamma)
        reweigts    = self.sigomid( gn_x * w_gamma )
        # Gate
        info_mask   = reweigts>=self.gate_treshold
        noninfo_mask= reweigts<self.gate_treshold
        x_1         = info_mask * x
        x_2         = noninfo_mask * x
        x           = self.reconstruct(x_1,x_2)
        return x
    
    def reconstruct(self,x_1,x_2):
        x_11,x_12 = torch.split(x_1, x_1.size(1)//2, dim=1)
        x_21,x_22 = torch.split(x_2, x_2.size(1)//2, dim=1)
        return torch.cat([ x_11+x_22, x_12+x_21 ],dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''
    def __init__(self, 
                 op_channel:int,
                 alpha:float = 1/2,
                 squeeze_radio:int = 2 ,
                 group_size:int = 2,
                 group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.up_channel     = up_channel   =   int(alpha*op_channel)
        self.low_channel    = low_channel  =   op_channel-up_channel
        self.squeeze1       = nn.Conv2d(up_channel,up_channel//squeeze_radio,kernel_size=1,bias=False)
        self.squeeze2       = nn.Conv2d(low_channel,low_channel//squeeze_radio,kernel_size=1,bias=False)
        #up
        self.GWC            = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=group_kernel_size, stride=1,padding=group_kernel_size//2, groups = group_size)
        self.PWC1           = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=1, bias=False)
        #low
        self.PWC2           = nn.Conv2d(low_channel//squeeze_radio, op_channel-low_channel//squeeze_radio,kernel_size=1, bias=False)
        self.advavg         = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        # Split
        up,low  = torch.split(x,[self.up_channel,self.low_channel],dim=1)
        up,low  = self.squeeze1(up),self.squeeze2(low)
        # Transform
        Y1      = self.GWC(up) + self.PWC1(up)
        Y2      = torch.cat( [self.PWC2(low), low], dim= 1 )
        # Fuse
        out     = torch.cat( [Y1,Y2], dim= 1 )
        out     = F.softmax( self.advavg(out), dim=1 ) * out
        out1,out2 = torch.split(out,out.size(1)//2,dim=1)
        return out1+out2


class ScConv(nn.Module):
    # https://github.com/cheng-haha/ScConv/blob/main/ScConv.py
    def __init__(self,
                op_channel:int,
                group_num:int = 16,
                gate_treshold:float = 0.5,
                alpha:float = 1/2,
                squeeze_radio:int = 2 ,
                group_size:int = 2,
                group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel, 
                       group_num            = group_num,  
                       gate_treshold        = gate_treshold)
        self.CRU = CRU(op_channel, 
                       alpha                = alpha, 
                       squeeze_radio        = squeeze_radio ,
                       group_size           = group_size ,
                       group_kernel_size    = group_kernel_size)
    
    def forward(self,x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x

class Bottleneck_ScConv(myBottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = myConv(c1, c_, k[0], 1)
        self.cv2 = ScConv(c2)

class C3_ScConv(myC3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_ScConv(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

######################################## ScConv end ########################################

# ---------------------------C3_EMSC start----------------------
from einops import rearrange

class EMSConv(nn.Module):
    # Efficient Multi-Scale Conv
    def __init__(self, channel=256, kernels=[3, 5]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = channel // 4
        assert min_ch >= 16, f'channel must Greater than {64}, but {channel}'

        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(Conv(c1=min_ch, c2=min_ch, k=ks))
        self.conv_1x1 = Conv(channel, channel, k=1)

    def forward(self, x):
        _, c, _, _ = x.size()
        x_cheap, x_group = torch.split(x, [c // 2, c // 2], dim=1)
        x_group = rearrange(x_group, 'bs (g ch) h w -> bs ch h w g', g=self.groups)
        x_group = torch.stack([self.convs[i](x_group[..., i]) for i in range(len(self.convs))])
        x_group = rearrange(x_group, 'g bs ch h w -> bs (g ch) h w')
        x = torch.cat([x_cheap, x_group], dim=1)
        x = self.conv_1x1(x)

        return x


class EMSConvP(nn.Module):
    # Efficient Multi-Scale Conv Plus
    def __init__(self, channel=256, kernels=[1, 3, 5, 7]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = channel // self.groups
        assert min_ch >= 16, f'channel must Greater than {16 * self.groups}, but {channel}'

        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(Conv(c1=min_ch, c2=min_ch, k=ks))
        self.conv_1x1 = Conv(channel, channel, k=1)

    def forward(self, x):
        x_group = rearrange(x, 'bs (g ch) h w -> bs ch h w g', g=self.groups)
        x_convs = torch.stack([self.convs[i](x_group[..., i]) for i in range(len(self.convs))])
        x_convs = rearrange(x_convs, 'g bs ch h w -> bs (g ch) h w')
        x_convs = self.conv_1x1(x_convs)

        return x_convs


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Bottleneck_EMSC(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = EMSConv(c2)


class C3_EMSC(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_EMSC(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


# ---------------------------C3_EMSC end----------------------

# ---------------------------C3_EMSCP start----------------------
class EMSConv(nn.Module):
    # Efficient Multi-Scale Conv
    def __init__(self, channel=256, kernels=[3, 5]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = channel // 4
        assert min_ch >= 16, f'channel must Greater than {64}, but {channel}'

        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(Conv(c1=min_ch, c2=min_ch, k=ks))
        self.conv_1x1 = Conv(channel, channel, k=1)

    def forward(self, x):
        _, c, _, _ = x.size()
        x_cheap, x_group = torch.split(x, [c // 2, c // 2], dim=1)
        x_group = rearrange(x_group, 'bs (g ch) h w -> bs ch h w g', g=self.groups)
        x_group = torch.stack([self.convs[i](x_group[..., i]) for i in range(len(self.convs))])
        x_group = rearrange(x_group, 'g bs ch h w -> bs (g ch) h w')
        x = torch.cat([x_cheap, x_group], dim=1)
        x = self.conv_1x1(x)

        return x


class EMSConvP(nn.Module):
    # Efficient Multi-Scale Conv Plus
    def __init__(self, channel=256, kernels=[1, 3, 5, 7]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = channel // self.groups
        assert min_ch >= 16, f'channel must Greater than {16 * self.groups}, but {channel}'

        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(Conv(c1=min_ch, c2=min_ch, k=ks))
        self.conv_1x1 = Conv(channel, channel, k=1)

    def forward(self, x):
        x_group = rearrange(x, 'bs (g ch) h w -> bs ch h w g', g=self.groups)
        x_convs = torch.stack([self.convs[i](x_group[..., i]) for i in range(len(self.convs))])
        x_convs = rearrange(x_convs, 'g bs ch h w -> bs (g ch) h w')
        x_convs = self.conv_1x1(x_convs)

        return x_convs


class Bottleneck_EMSC(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = EMSConv(c2)


class Bottleneck_EMSCP(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = EMSConvP(c2)


class C3_EMSCP(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_EMSCP(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

# ---------------------------C3_EMSCP end----------------------

# ---------------------  C3_PKIModule start--------------------------------------
class GSiLU(nn.Module):
    """Global Sigmoid-Gated Linear Unit, reproduced from paper <SIMPLE CNN FOR VISION>"""

    def __init__(self):
        super().__init__()
        self.adpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return x * torch.sigmoid(self.adpool(x))


class PKIModule_CAA(nn.Module):
    def __init__(self, ch, h_kernel_size=11, v_kernel_size=11) -> None:
        super().__init__()

        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(ch, ch)
        self.h_conv = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, h_kernel_size // 2), 1, ch)
        self.v_conv = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), 1, ch)
        self.conv2 = Conv(ch, ch)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor

def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

class PKIModule(nn.Module):
    def __init__(self, inc, ouc, kernel_sizes=(3, 5, 7, 9, 11), expansion=1.0, with_caa=True, caa_kernel_size=11,
                 add_identity=True) -> None:
        super().__init__()
        hidc = make_divisible(int(ouc * expansion), 8)

        self.pre_conv = Conv(inc, hidc)
        self.dw_conv = nn.ModuleList(
            nn.Conv2d(hidc, hidc, kernel_size=k, padding=autopad(k), groups=hidc) for k in kernel_sizes)
        self.pw_conv = Conv(hidc, hidc)
        self.post_conv = Conv(hidc, ouc)

        if with_caa:
            self.caa_factor = PKIModule_CAA(hidc, caa_kernel_size, caa_kernel_size)
        else:
            self.caa_factor = None

        self.add_identity = add_identity and inc == ouc

    def forward(self, x):
        x = self.pre_conv(x)

        y = x
        x = self.dw_conv[0](x)
        x = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv[1:]], dim=0), dim=0)
        x = self.pw_conv(x)

        if self.caa_factor is not None:
            y = self.caa_factor(y)
        if self.add_identity:
            y = x * y
            x = x + y
        else:
            x = x * y

        x = self.post_conv(x)
        return x


class C3_PKIModule(C3):
    def __init__(self, c1, c2, n=1, kernel_sizes=(3, 5, 7, 9, 11), expansion=1.0, with_caa=True, caa_kernel_size=11,
                 add_identity=True, g=1, e=0.5):
        super().__init__(c1, c2, n, True, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(
            *(PKIModule(c_, c_, kernel_sizes, expansion, with_caa, caa_kernel_size, add_identity) for _ in range(n)))


# ----------------------  C3_PKIModule end--------------------------------------

# ---------------------------C3_DWR start----------------------------
class DWR(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv_3x3 = myConv(dim, dim // 2, 3)

        self.conv_3x3_d1 = myConv(dim // 2, dim, 3, d=1)
        self.conv_3x3_d3 = myConv(dim // 2, dim // 2, 3, d=3)
        self.conv_3x3_d5 = myConv(dim // 2, dim // 2, 3, d=5)

        self.conv_1x1 = myConv(dim * 2, dim, k=1)

    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)
        x1, x2, x3 = self.conv_3x3_d1(conv_3x3), self.conv_3x3_d3(conv_3x3), self.conv_3x3_d5(conv_3x3)
        x_out = torch.cat([x1, x2, x3], dim=1)
        x_out = self.conv_1x1(x_out) + x
        return x_out


class C3_DWR(myC3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DWR(c_) for _ in range(n)))
# ---------------------------C3_DWR end----------------------------

#----------------------------C3_DWR 改进---------------------------
# ----------------------------------------------------------------
# DWRv1 参考部分卷积的思想
class DWRv1(nn.Module):
    def __init__(self, dim, n_div=2, forward='split_cat') -> None:
        super().__init__()
        # 计算被部分卷积处理的通道数
        self.dim_conv3 = dim // n_div
        # 计算未被处理的通道数
        self.dim_untouched = dim - self.dim_conv3

        self.conv_3x3 = myConv(self.dim_conv3, self.dim_conv3 // 2, 3)

        self.conv_3x3_d1 = myConv(self.dim_conv3 // 2, self.dim_conv3, 3, d=1)
        self.conv_3x3_d3 = myConv(self.dim_conv3 // 2, self.dim_conv3 // 2, 3, d=3)
        self.conv_3x3_d5 = myConv(self.dim_conv3 // 2, self.dim_conv3 // 2, 3, d=5)

        self.conv_1x1 = myConv(self.dim_conv3 * 2, self.dim_conv3, k=1)

        self.forward_mode = forward

    def forward(self, x):
        # 拆分输入特征图
        x_conv, x_untouched = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)

        conv_3x3 = self.conv_3x3(x_conv)
        x1, x2, x3 = self.conv_3x3_d1(conv_3x3), self.conv_3x3_d3(conv_3x3), self.conv_3x3_d5(conv_3x3)
        x_conv_out = torch.cat([x1, x2, x3], dim=1)
        x_conv_out = self.conv_1x1(x_conv_out) + x_conv

        # 合并处理后的部分和未处理的部分
        x_out = torch.cat([x_conv_out, x_untouched], dim=1)
        return x_out


class C3_DWRv1(myC3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DWRv1(c_) for _ in range(n)))

# ----------------------------------------------------------------
# DWRv2 引入GSConv
class DWRv2(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv_3x3 = GSConv(dim, dim // 2, 3)

        self.conv_3x3_d1 = myConv(dim // 2, dim, 3, d=1)
        self.conv_3x3_d3 = myConv(dim // 2, dim // 2, 3, d=3)
        self.conv_3x3_d5 = myConv(dim // 2, dim // 2, 3, d=5)

        self.conv_1x1 = GSConv(dim * 2, dim, k=1)

    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)
        x1, x2, x3 = self.conv_3x3_d1(conv_3x3), self.conv_3x3_d3(conv_3x3), self.conv_3x3_d5(conv_3x3)
        x_out = torch.cat([x1, x2, x3], dim=1)
        x_out = self.conv_1x1(x_out) + x
        return x_out


class C3_DWRv2(myC3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DWRv2(c_) for _ in range(n)))

# ----------------------------------------------------------------
# DWRv3 通过 Global Average Pooling 提取全局上下文信息并进行融合
class DWRv3(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv_3x3 = myConv(dim, dim // 2, 3)

        # Define three dilation branches
        self.conv_3x3_d1 = myConv(dim // 2, dim, 3, d=1)
        self.conv_3x3_d3 = myConv(dim // 2, dim // 2, 3, d=3)
        self.conv_3x3_d5 = myConv(dim // 2, dim // 2, 3, d=5)

        # 1x1 convolution to fuse features
        self.conv_1x1 = myConv(dim * 2 + dim // 2, dim, k=1)

        # Global Average Pooling path
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # GAP to 1x1
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, bias=False),  # Reduce channels
            nn.SiLU(),  # Activation
            nn.Conv2d(dim // 2, dim // 2, kernel_size=1, bias=False),  # Restore channels
        )

    def forward(self, x):
        """Forward pass with global context fusion."""
        # Main convolution path
        conv_3x3 = self.conv_3x3(x)
        x1 = self.conv_3x3_d1(conv_3x3)
        x2 = self.conv_3x3_d3(conv_3x3)
        x3 = self.conv_3x3_d5(conv_3x3)

        # Global Average Pooling path
        gap = self.global_pool(x)  # Global context
        gap = self.fc(gap)  # Transform global context

        # Upsample GAP output to match spatial size
        gap = F.interpolate(gap, size=x1.shape[-2:], mode="bilinear", align_corners=False)

        # Concatenate all branches and GAP
        x_out = torch.cat([x1, x2, x3, gap], dim=1)

        # Final fusion
        x_out = self.conv_1x1(x_out) + x
        return x_out


class C3_DWRv3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = myConv(c1, c_, 1, 1)
        self.cv2 = myConv(c1, c_, 1, 1)
        self.cv3 = myConv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(DWRv3(c_) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

 # ----------------------------------------------------------------   
# DWRv4 使用动态加权机制对多分支特征进行加权融合(两种版本)
class AdaptiveDilationBlock(nn.Module):
    """Block with adaptive dilation rates based on input features."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv_3x3 = myConv(dim, dim // 2, 3)
        
        # Base convolutions with different dilation rates
        self.conv_3x3_d1 = myConv(dim // 2, dim // 2, 3, d=1)
        self.conv_3x3_d3 = myConv(dim // 2, dim // 2, 3, d=3)
        self.conv_3x3_d5 = myConv(dim // 2, dim // 2, 3, d=5)
        
        # Attention mechanism to learn dilation weights
        self.dilation_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim // 2, 3, 1),  # 3 outputs for 3 dilation rates
            nn.Softmax(dim=1)
        )
        
        # Adjust the output channels to match the expected dim
        self.conv_1x1 = myConv((dim // 2) * 3, dim, k=1)  # Changed from dim*2 to (dim//2)*3

    def forward(self, x):
        # Initial feature transformation
        conv_3x3 = self.conv_3x3(x)
        
        # Get features from different dilation branches
        x1 = self.conv_3x3_d1(conv_3x3)
        x2 = self.conv_3x3_d3(conv_3x3)
        x3 = self.conv_3x3_d5(conv_3x3)
        
        # Compute attention weights for each dilation branch
        attention_weights = self.dilation_attention(conv_3x3)  # shape: [B, 3, 1, 1]
        
        # Apply attention weights to each branch
        x1_weighted = x1 * attention_weights[:, 0:1, :, :]
        x2_weighted = x2 * attention_weights[:, 1:2, :, :]
        x3_weighted = x3 * attention_weights[:, 2:3, :, :]
        
        # Concatenate all features
        x_out = torch.cat([x1_weighted, x2_weighted, x3_weighted], dim=1)
        
        # Final transformation with residual connection
        x_out = self.conv_1x1(x_out) + x
        return x_out

class AdaptiveDWR(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv_3x3 = myConv(dim, dim // 2, 3)

        self.conv_3x3_d1 = myConv(dim // 2, dim, 3, d=1)
        self.conv_3x3_d3 = myConv(dim // 2, dim // 2, 3, d=3)
        self.conv_3x3_d5 = myConv(dim // 2, dim // 2, 3, d=5)

        self.conv_1x1 = myConv(dim * 2, dim, k=1)

        # 定义可学习的权重
        self.weights = nn.Parameter(torch.ones(3))

    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)
        x1, x2, x3 = self.conv_3x3_d1(conv_3x3), self.conv_3x3_d3(conv_3x3), self.conv_3x3_d5(conv_3x3)

        # 对权重进行 softmax 归一化
        weights = torch.softmax(self.weights, dim=0)

        # 加权拼接
        weighted_x1 = weights[0] * x1
        weighted_x2 = weights[1] * x2
        weighted_x3 = weights[2] * x3

        x_out = torch.cat([weighted_x1, weighted_x2, weighted_x3], dim=1)
        x_out = self.conv_1x1(x_out) + x
        return x_out

class C3_DWRv4(myC3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(AdaptiveDilationBlock(c_) for _ in range(n)))

  
# ----------------------------------------------------------------
# DWRv5 结合DWR和Bottleneck结构
class C3_DWRv5(myC3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m_dwr = nn.Sequential(*(DWR(c_ // 2) for _ in range(n)))
        self.m_bottleneck = nn.Sequential(*(myBottleneck(c_ // 2, c_ // 2, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        cv1_output = self.cv1(x)
        # 沿着通道维度分割成两部分
        cv1_part1, cv1_part2 = torch.chunk(cv1_output, 2, dim=1)
        dwr_output = self.m_dwr(cv1_part1)
        bottleneck_output = self.m_bottleneck(cv1_part2)
        combined_output = torch.cat([dwr_output, bottleneck_output], dim=1)
        return self.cv3(torch.cat((combined_output, self.cv2(x)), 1))

# ----------------------------------------------------------------
# DWRv6 结合DWR和GSBottleneck结构
class GSBottleneck(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__()
        c_ = int(c2*e)
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 3, 1, act=False))
        self.shortcut = Conv(c1, c2, 1, 1, act=False)
 
    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)

class C3_DWRv6(myC3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m_dwr = nn.Sequential(*(DWR(c_ // 2) for _ in range(n)))
        self.gsb = nn.Sequential(*(GSBottleneck(c_ // 2, c_ // 2, e=1.0) for _ in range(n)))

    def forward(self, x):
        cv1_output = self.cv1(x)
        # 沿着通道维度分割成两部分
        cv1_part1, cv1_part2 = torch.chunk(cv1_output, 2, dim=1)
        dwr_output = self.m_dwr(cv1_part1)
        bottleneck_output = self.gsb(cv1_part2)
        combined_output = torch.cat([dwr_output, bottleneck_output], dim=1)
        return self.cv3(torch.cat((combined_output, self.cv2(x)), 1))

class C3_DWRv6_split(myC3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m_dwr = nn.Sequential(*(DWR(int(c_ * 3 / 4)) for _ in range(n)))
        self.gsb = nn.Sequential(*(GSBottleneck(int(c_ / 4), int(c_ / 4), e=1.0) for _ in range(n)))

    def forward(self, x):
        cv1_output = self.cv1(x)
        # 计算分割点
        split_point = int(cv1_output.shape[1] * 3 / 4)
        # 沿着通道维度分割成两部分
        cv1_part1, cv1_part2 = cv1_output[:, :split_point, :, :], cv1_output[:, split_point:, :, :]
        dwr_output = self.m_dwr(cv1_part1)
        bottleneck_output = self.gsb(cv1_part2)
        combined_output = torch.cat([dwr_output, bottleneck_output], dim=1)
        return self.cv3(torch.cat((combined_output, self.cv2(x)), 1))
#----------------------------------------------------------------------------------
# DWRv7 结合DWR和SimAM结构
class SimAM(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
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

        return x * self.activaton(y)

class DWR_SimAM(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv_3x3 = myConv(dim, dim // 2, 3)

        self.conv_3x3_d1 = myConv(dim // 2, dim, 3, d=1)
        self.conv_3x3_d3 = myConv(dim // 2, dim // 2, 3, d=3)
        self.conv_3x3_d5 = myConv(dim // 2, dim // 2, 3, d=5)

        self.conv_1x1 = myConv(dim * 2, dim, k=1)
        self.simam = SimAM(dim)

    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)
        x1, x2, x3 = self.conv_3x3_d1(conv_3x3), self.conv_3x3_d3(conv_3x3), self.conv_3x3_d5(conv_3x3)
        x_out = torch.cat([x1, x2, x3], dim=1)
        x_out = self.simam(x_out)
        x_out = self.conv_1x1(x_out) + x
        return x_out

class C3_DWRv7(myC3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DWR_SimAM(c_) for _ in range(n)))