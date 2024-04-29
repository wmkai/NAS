# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# adapted from OFA: https://github.com/mit-han-lab/once-for-all

from collections import OrderedDict
import copy
from turtle import forward
from cv2 import split

import torch
import torch.nn as nn
import torch.nn.functional as F

from .static_layers import MBInvertedConvLayer, ConvBnActLayer, LinearLayer, SELayer, ShortcutLayer, h_sigmoid, InjectionMultiSum
from .dynamic_ops import DynamicBatchNorm1dWithIdx, DynamicSeparableConv2d, DynamicPointConv2d, DynamicBatchNorm2d, DynamicLinear, DynamicLinearWithIdx, DynamicSE
from .nn_utils import int2list, get_net_device, copy_bn, copy_bn_idx, build_activation, make_divisible, compute_idx_from_channel_split

# 包括DynamicMBConvLayer、DynamicConvBnActLayer、DynamicLinearLayer、DynamicShortcutLayer
# 这些类都包含forward和get_active_subnet方法

class DynamicMBConvLayer(nn.Module):
    
    def __init__(self, in_channel_list, out_channel_list,
                 kernel_size_list=3, expand_ratio_list=6, stride=1, act_func='relu6', use_se=False, channels_per_group=1):
        super(DynamicMBConvLayer, self).__init__()
        
        self.in_channel_list = int2list(in_channel_list)
        self.out_channel_list = int2list(out_channel_list)
        
        self.kernel_size_list = int2list(kernel_size_list, 1)
        self.expand_ratio_list = int2list(expand_ratio_list, 1)
        
        self.stride = stride
        self.act_func = act_func
        self.use_se = use_se
        self.channels_per_group = channels_per_group
        
        # build modules
        max_middle_channel = round(max(self.in_channel_list) * max(self.expand_ratio_list))
        if max(self.expand_ratio_list) == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([ # pw
                ('conv', DynamicPointConv2d(max(self.in_channel_list), max_middle_channel)),
                ('bn', DynamicBatchNorm2d(max_middle_channel)),
                ('act', build_activation(self.act_func, inplace=True)),
            ]))
        
        self.depth_conv = nn.Sequential(OrderedDict([ # dw
            ('conv', DynamicSeparableConv2d(max_middle_channel, self.kernel_size_list, stride=self.stride, channels_per_group=self.channels_per_group)),
            ('bn', DynamicBatchNorm2d(max_middle_channel)),
            ('act', build_activation(self.act_func, inplace=True))
        ]))
        if self.use_se:
            self.depth_conv.add_module('se', DynamicSE(max_middle_channel))
        
        self.point_linear = nn.Sequential(OrderedDict([ # pw
            ('conv', DynamicPointConv2d(max_middle_channel, max(self.out_channel_list))),
            ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
        ]))
        
        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        in_channel = x.size(1)
        
        if self.inverted_bottleneck is not None:
            self.inverted_bottleneck.conv.active_out_channel = \
                make_divisible(round(in_channel * self.active_expand_ratio), 8)

        self.depth_conv.conv.active_kernel_size = self.active_kernel_size
        self.point_linear.conv.active_out_channel = self.active_out_channel
        
        if self.inverted_bottleneck is not None:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x
    

    def get_active_subnet(self, in_channel, preserve_weight=True):
        middle_channel = make_divisible(round(in_channel * self.active_expand_ratio), 8)
        channels_per_group = self.depth_conv.conv.channels_per_group

        # build the new layer
        sub_layer = MBInvertedConvLayer(
            in_channel, self.active_out_channel, self.active_kernel_size, self.stride, self.active_expand_ratio,
            act_func=self.act_func, mid_channels=middle_channel, use_se=self.use_se, channels_per_group=channels_per_group
        )
        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            return sub_layer

        # copy weight from current layer
        if sub_layer.inverted_bottleneck is not None:
            sub_layer.inverted_bottleneck.conv.weight.data.copy_(
                self.inverted_bottleneck.conv.conv.weight.data[:middle_channel, :in_channel, :, :]
            )
            copy_bn(sub_layer.inverted_bottleneck.bn, self.inverted_bottleneck.bn.bn)

        sub_layer.depth_conv.conv.weight.data.copy_(
            self.depth_conv.conv.get_active_filter(middle_channel, self.active_kernel_size).data
        )

        copy_bn(sub_layer.depth_conv.bn, self.depth_conv.bn.bn)

        if self.use_se:
            se_mid = make_divisible(middle_channel // SELayer.REDUCTION, divisor=8)
            sub_layer.depth_conv.se.fc.reduce.weight.data.copy_(
                self.depth_conv.se.fc.reduce.weight.data[:se_mid, :middle_channel, :, :]
            )
            sub_layer.depth_conv.se.fc.reduce.bias.data.copy_(self.depth_conv.se.fc.reduce.bias.data[:se_mid])

            sub_layer.depth_conv.se.fc.expand.weight.data.copy_(
                self.depth_conv.se.fc.expand.weight.data[:middle_channel, :se_mid, :, :]
            )
            sub_layer.depth_conv.se.fc.expand.bias.data.copy_(self.depth_conv.se.fc.expand.bias.data[:middle_channel])

        sub_layer.point_linear.conv.weight.data.copy_(
            self.point_linear.conv.conv.weight.data[:self.active_out_channel, :middle_channel, :, :]
        )
        copy_bn(sub_layer.point_linear.bn, self.point_linear.bn.bn)

        return sub_layer


class DynamicConvBnActLayer(nn.Module):
    
    def __init__(self, in_channel_list, out_channel_list, kernel_size=3, stride=1, dilation=1,
                 use_bn=True, act_func='relu6'):
        super(DynamicConvBnActLayer, self).__init__()
        
        self.in_channel_list = int2list(in_channel_list)
        self.out_channel_list = int2list(out_channel_list)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bn = use_bn
        self.act_func = act_func
        
        self.conv = DynamicPointConv2d(
            max_in_channels=max(self.in_channel_list), max_out_channels=max(self.out_channel_list),
            kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation,
        )
        if self.use_bn:
            self.bn = DynamicBatchNorm2d(max(self.out_channel_list))

        if self.act_func is not None:
            self.act = build_activation(self.act_func, inplace=True)
        
        self.active_out_channel = max(self.out_channel_list)
    
    def forward(self, x):
        self.conv.active_out_channel = self.active_out_channel
        
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.act_func is not None:
            x = self.act(x)
        return x
    
    def get_active_subnet(self, in_channel, preserve_weight=True):
        sub_layer = ConvBnActLayer(
            in_channel, self.active_out_channel, self.kernel_size, self.stride, self.dilation,
            use_bn=self.use_bn, act_func=self.act_func
        )
        sub_layer = sub_layer.to(get_net_device(self))
        
        if not preserve_weight:
            return sub_layer
        
        sub_layer.conv.weight.data.copy_(self.conv.conv.weight.data[:self.active_out_channel, :in_channel, :, :])
        if self.use_bn:
            copy_bn(sub_layer.bn, self.bn.bn)
        
        return sub_layer        

class DynamicLinearLayer(nn.Module):

    def __init__(self, in_features_list, out_features, bias=True):
        super(DynamicLinearLayer, self).__init__()
        
        self.in_features_list = int2list(in_features_list)
        self.out_features = out_features
        self.bias = bias
        #self.dropout_rate = dropout_rate
        #
        #if self.dropout_rate > 0:
        #    self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        #else:
        #    self.dropout = None
        self.linear = DynamicLinear(
            max_in_features=max(self.in_features_list), max_out_features=self.out_features, bias=self.bias
        )
    
    def forward(self, x):
        #if self.dropout is not None:
        #    x = self.dropout(x)
        return self.linear(x)

    def get_active_subnet(self, in_features, preserve_weight=True):
        #sub_layer = LinearLayer(in_features, self.out_features, self.bias, dropout_rate=self.dropout_rate)
        sub_layer = LinearLayer(in_features, self.out_features, self.bias)
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer
        
        sub_layer.linear.weight.data.copy_(self.linear.linear.weight.data[:self.out_features, :in_features])
        if self.bias:
            sub_layer.linear.bias.data.copy_(self.linear.linear.bias.data[:self.out_features])
        return sub_layer



class DynamicShortcutLayer(nn.Module):
    
    def __init__(self, in_channel_list, out_channel_list, reduction=1):
        super(DynamicShortcutLayer, self).__init__()
        
        self.in_channel_list = int2list(in_channel_list)
        self.out_channel_list = int2list(out_channel_list)
        self.reduction = reduction
        
        self.conv = DynamicPointConv2d(
            max_in_channels=max(self.in_channel_list), max_out_channels=max(self.out_channel_list),
            kernel_size=1, stride=1,
        )

        self.active_out_channel = max(self.out_channel_list)
    
    def forward(self, x):
        in_channel = x.size(1)

        #identity mapping
        if in_channel == self.active_out_channel and self.reduction == 1:
            return x
        #average pooling, if size doesn't match
        if self.reduction > 1:
            padding = 0 if x.size(-1) % 2 == 0 else 1 # 训练时图片被resize成正方形，所以最后两个维度是一样的，只要判断一个
            x = F.avg_pool2d(x, self.reduction, padding=padding)

        #1*1 conv, if #channels doesn't match
        if in_channel != self.active_out_channel:
            self.conv.active_out_channel = self.active_out_channel
            x = self.conv(x)
        return x
    
    def get_active_subnet(self, in_channel, preserve_weight=True):
        sub_layer = ShortcutLayer(
            in_channel, self.active_out_channel, self.reduction 
        )
        sub_layer = sub_layer.to(get_net_device(self))
        
        if not preserve_weight:
            return sub_layer
        
        sub_layer.conv.weight.data.copy_(self.conv.conv.weight.data[:self.active_out_channel, :in_channel, :, :])
        
        return sub_layer

class DynamicInjectionMultiSum(nn.Module): # 默认用这个
    def __init__(
        self,
        inp_channel_list: list,
        oup: int,
        norm_cfg=dict(type='BN', requires_grad=True),
        activations = None, # 这里是摆设，self.act = h_sigmoid()
    ) -> None:
        super(DynamicInjectionMultiSum, self).__init__()
        self.inp_channel_list = int2list(inp_channel_list)
        self.oup = oup
        self.norm_cfg = norm_cfg

        if activations is None:
            activations = nn.ReLU6
        elif activations == 'relu6':
            activations = nn.ReLU6
        elif activations == 'relu':
            activations = nn.ReLU
        elif activations == 'hswish':
            activations = nn.Hardswish
        elif activations == 'gelu':
            activations = nn.GELU
        else:
            raise NotImplementedError

        self.local_embedding = nn.Sequential(OrderedDict([
            ('conv', DynamicPointConv2d(
                max_in_channels=max(self.inp_channel_list), max_out_channels=self.oup,
                kernel_size=1, stride=1, dilation=1,
            )),
            ('norm', DynamicBatchNorm2d(self.oup))
        ]))

        self.global_embedding = nn.Sequential(OrderedDict([
            ('conv', DynamicPointConv2d(
                max_in_channels=max(self.inp_channel_list), max_out_channels=self.oup,
                kernel_size=1, stride=1, dilation=1,
            )),
            ('norm', DynamicBatchNorm2d(self.oup))
        ]))

        self.global_act = nn.Sequential(OrderedDict([
            ('conv', DynamicPointConv2d(
                max_in_channels=max(self.inp_channel_list), max_out_channels=self.oup,
                kernel_size=1, stride=1, dilation=1,
            )),
            ('norm', DynamicBatchNorm2d(self.oup))
        ]))
        
        # self.local_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        # self.global_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        # self.global_act = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid()

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

    def get_active_subnet(self, inp, preserve_weight=True):
        sub_layer = InjectionMultiSum(
            inp, self.oup, self.norm_cfg, 
        )
        sub_layer = sub_layer.to(get_net_device(self))
        
        if not preserve_weight:
            return sub_layer
        
        # sub_layer.conv.weight.data.copy_(self.conv.conv.weight.data[:self.active_out_channel, :in_channel, :, :])
        sub_layer.local_embedding.conv.weight.data.copy_(self.local_embedding.conv.conv.weight.data[:, :inp, :, :])
        sub_layer.global_embedding.conv.weight.data.copy_(self.global_embedding.conv.conv.weight.data[:, :inp, :, :])
        sub_layer.global_act.conv.weight.data.copy_(self.global_act.conv.conv.weight.data[:, :inp, :, :])
        return sub_layer

class DynamicClsHead(nn.Module):
    def __init__(self, in_feature_list, max_channel_split: list, out_features=1000, bias=True):
        super(DynamicClsHead, self).__init__()
        
        self.in_feature_list = int2list(in_feature_list)
        self.max_channel_split = max_channel_split
        # 通过每部分channel的个数，得到每部分开始的索引
        self.max_channel_start_idx = compute_idx_from_channel_split(self.max_channel_split)
        self.out_features = out_features
        self.bias = bias
        
        # build modules        
        self.bn = DynamicBatchNorm1dWithIdx(max_feature_dim=max(self.in_feature_list))
        self.l = DynamicLinearWithIdx(max_in_features=max(self.in_feature_list), max_out_features=out_features, bias=self.bias)

        # 新增
        self.active_channel_split = max_channel_split

    def forward(self, x):
        assert len(self.active_channel_split) == len(self.max_channel_split)        
        split_channel_idx = []
        for i in range(len(self.max_channel_start_idx)):
            for j in range(self.max_channel_start_idx[i], self.max_channel_start_idx[i]+self.active_channel_split[i]):
                split_channel_idx.append(j)
        
        self.bn.active_idx = split_channel_idx
        self.l.active_in_idx = split_channel_idx

        return self.l(self.bn(x))

    def get_active_subnet(self, in_features, preserve_weight=True):
        assert len(self.active_channel_split) == len(self.max_channel_split)        
        split_channel_idx = []
        for i in range(len(self.max_channel_start_idx)):
            for j in range(self.max_channel_start_idx[i], self.max_channel_start_idx[i]+self.active_channel_split[i]):
                split_channel_idx.append(j)
        assert in_features == len(split_channel_idx)
        #sub_layer = LinearLayer(in_features, self.out_features, self.bias, dropout_rate=self.dropout_rate)
        sub_layer = nn.Sequential(OrderedDict([
          ('bn', nn.BatchNorm1d(in_features)),
          ('l', LinearLayer(in_features, self.out_features, self.bias)),
        ]))
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer
        
        temp_linear_weight = self.l.linear.weight.data[:self.out_features, :]
        sub_layer.l.linear.weight.data.copy_(temp_linear_weight[:, split_channel_idx])
        if self.bias:
            sub_layer.l.linear.bias.data.copy_(self.l.linear.bias.data[:self.out_features])
        copy_bn_idx(sub_layer.bn, self.bn.bn, split_channel_idx)
        return sub_layer

    



