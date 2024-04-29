# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# adapted from OFA: https://github.com/mit-han-lab/once-for-all

from collections import OrderedDict
from typing_extensions import runtime
import torch.nn as nn
from .nn_utils import get_same_padding, build_activation, make_divisible, drop_path, get_shape, drop_connect
from .activations import *
from mmcv.cnn import build_norm_layer
from mmcv.cnn import ConvModule

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class PyramidPoolAgg(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        # import pdb; pdb.set_trace()
        B, C, H, W = get_shape(inputs[-1])
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        return torch.cat([nn.functional.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class InjectionMultiSum(nn.Module): # 默认用这个
    def __init__(
        self,
        inp: int,
        oup: int,
        norm_cfg=dict(type='BN', requires_grad=True)
    ) -> None:
        super(InjectionMultiSum, self).__init__()
        self.norm_cfg = norm_cfg

        self.local_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_act = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid() # 激活函数

    def forward(self, x_l, x_g):
        '''
        x_g: global features
        x_l: local features
        '''
        # pdb.set_trace()
        B, C, H, W = x_l.shape
        local_feat = self.local_embedding(x_l) # 1x1 conv
        
        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
        
        global_feat = self.global_embedding(x_g) # 1x1 conv
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False) # 上采样到左边传来的局部特征一样的尺寸
        
        out = local_feat * sig_act + global_feat
        return out


class SELayer(nn.Module):
    REDUCTION = 4

    def __init__(self, channel):
        super(SELayer, self).__init__()

        self.channel = channel
        self.reduction = SELayer.REDUCTION

        num_mid = make_divisible(self.channel // self.reduction, divisor=8)

        self.fc = nn.Sequential(OrderedDict([
                            ('reduce', nn.Conv2d(self.channel, num_mid, 1, 1, 0, bias=True)),
                            ('relu', nn.ReLU(inplace=True)),
                            ('expand', nn.Conv2d(num_mid, self.channel, 1, 1, 0, bias=True)),
                            ('h_sigmoid', Hsigmoid(inplace=True)),
        ]))

    def forward(self, x):
        #x: N, C, H, W 
        y = x.mean(3, keepdim=True).mean(2, keepdim=True) # N, C, 1, 1
        y = self.fc(y)
        return x * y


class ConvBnActLayer(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, dilation=1, groups=1, bias=False,
                 use_bn=True, act_func='relu'):
        super(ConvBnActLayer, self).__init__()
        # default normal 3x3_Conv with bn and relu
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.use_bn = use_bn
        self.act_func = act_func

        pad = get_same_padding(self.kernel_size)
        self.conv = nn.Conv2d(in_channels, out_channels, self.kernel_size, 
            stride, pad, dilation=dilation, groups=groups, bias=bias
        )
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)

        self.act = build_activation(self.act_func, inplace=True)
        

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x


class IdentityLayer(nn.Module):

    def __init__(self, ):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x

class LinearLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        #self.dropout_rate = dropout_rate
        #if self.dropout_rate > 0:
        #    self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        #else:
        #    self.dropout = None

        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        #if dropout is not None:
        #    x = self.dropout(x)
        return self.linear(x)



class ShortcutLayer(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=1):
        super(ShortcutLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.reduction = reduction

        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)


    def forward(self, x):
        if self.reduction > 1:
            # padding = 0 if x.size(-1) % 2 == 0 else 1
            # x = F.avg_pool2d(x, self.reduction, padding=padding)
            # 测试时图片没有被resize成正方形，所以最后两个维度是不一样的，要判断两个
            padding1 = 0 if x.size(-2) % 2 == 0 else 1
            padding2 = 0 if x.size(-1) % 2 == 0 else 1
            x = F.avg_pool2d(x, self.reduction, padding=(padding1, padding2))
        if self.in_channels != self.out_channels:
            x = self.conv(x)
        return x



class MBInvertedConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, expand_ratio=6, mid_channels=None, act_func='relu6', use_se=False, channels_per_group=1):
        super(MBInvertedConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.act_func = act_func
        self.use_se = use_se

        self.channels_per_group = channels_per_group

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self.in_channels, feature_dim, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(feature_dim)),
                ('act', build_activation(self.act_func, inplace=True)),
            ]))

        assert feature_dim % self.channels_per_group == 0
        active_groups = feature_dim // self.channels_per_group
        pad = get_same_padding(self.kernel_size)
        depth_conv_modules = [
            ('conv', nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, pad, groups=active_groups, bias=False)),
            ('bn', nn.BatchNorm2d(feature_dim)),
            ('act', build_activation(self.act_func, inplace=True))
        ]
        if self.use_se:
            depth_conv_modules.append(('se', SELayer(feature_dim)))
        self.depth_conv = nn.Sequential(OrderedDict(depth_conv_modules))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
        ]))

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x



class MobileInvertedResidualBlock(nn.Module):

    # def __init__(self, mobile_inverted_conv, shortcut):
    def __init__(self, mobile_inverted_conv):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        # self.shortcut = shortcut
        self.use_shortcut = True

    def forward(self, x):
        # if self.mobile_inverted_conv is None: # or isinstance(self.mobile_inverted_conv, ZeroLayer):
        #     res = x
        # elif self.shortcut is None: # or isinstance(self.shortcut, ZeroLayer):
        #     res = self.mobile_inverted_conv(x)
        # else: # 都非空
        #     im = self.shortcut(x) # shortcut是1x1 conv不改变大小
        #     x = self.mobile_inverted_conv(x) 
        #     res = x + im
        # return res

        if self.use_shortcut:
            return x + self.mobile_inverted_conv(x)
        else:
            return self.mobile_inverted_conv(x)

# 新增
class TransformerBlock(nn.Module):

    # def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
    #              drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True)):
    def __init__(self, attn, mlp, drop_path=0.):
        super(TransformerBlock, self).__init__()

        self.attn = attn
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path)# if drop_path > 0. else nn.Identity()
        self.mlp = mlp        

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1

'''**************** for transformer ****************'''

class StaticMLP(nn.Module):
    def __init__(self, fc1, dwconv, act, fc2, drop):
        super(StaticMLP, self).__init__()

        self.fc1 = fc1
        self.dwconv = dwconv
        self.act = act
        self.fc2 = fc2
        self.drop = drop
    
    def forward(self, x):
        x = self.fc1(x) # conv实现 [B, 768, 8, 8]
        x = self.dwconv(x) # [B, 768, 8, 8]
        x = self.act(x) # [B, 768, 8, 8]
        x = self.drop(x) # [B, 768, 8, 8]
        x = self.fc2(x) # conv实现 [B, 384, 8, 8]
        x = self.drop(x) # [B, 384, 8, 8]
        return x

class StaticAttention(nn.Module):
    def __init__(self, to_q, to_k, to_v, proj, key_dim, num_heads, attn_ratio):
        super(StaticAttention, self).__init__()

        self.to_q = to_q
        self.to_k = to_k
        self.to_v = to_v
        self.proj = proj
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.attn_ratio = attn_ratio

    
    def forward(self, x):
        B, C, H, W = x.shape
        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2) # [B, self.num_heads = 8, H * W = 8 x 8, self.key_dim = 16]
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W) # 同qq
        vv = self.to_v(x).reshape(B, self.num_heads, int(self.key_dim * self.attn_ratio), H * W).permute(0, 1, 3, 2) # [B, self.num_heads = 8, H * W = 8 x 8, self.d = 32]

        attn = torch.matmul(qq, kk) # [B, 8, 64, 64]
        attn = attn.softmax(dim=-1) # dim = k

        xx = torch.matmul(attn, vv) # [B, 8, 64, 32]
        
        xx = xx.permute(0, 1, 3, 2).reshape(B, int(self.key_dim * self.attn_ratio) * self.num_heads, H, W) # [B, 256, 8, 8]
        xx = self.proj(xx) # [B, 384, 8, 8]
        return xx




