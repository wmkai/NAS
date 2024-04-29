# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# adapted from OFA: https://github.com/mit-han-lab/once-for-all

from torch.autograd.function import Function
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch
from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm
import torch.distributed as dist

from .nn_utils import get_same_padding, make_divisible, sub_filter_start_end, compute_idx_from_channel_split
from .static_layers import SELayer

# 包括DynamicSeparableConv2d、DynamicPointConv2d、DynamicLinear、DynamicSE、DynamicBatchNorm2d，
# 都包含forward方法

class DynamicSeparableConv2d(nn.Module):
    KERNEL_TRANSFORM_MODE = None  # None or 1
    
    def __init__(self, max_in_channels, kernel_size_list, stride=1, dilation=1, channels_per_group=1, bias=False):
        super(DynamicSeparableConv2d, self).__init__()
        
        self.max_in_channels = max_in_channels
        self.channels_per_group = channels_per_group
        assert self.max_in_channels % self.channels_per_group == 0
        self.kernel_size_list = kernel_size_list
        self.stride = stride
        self.dilation = dilation
        
        self.conv = nn.Conv2d(
            self.max_in_channels, self.max_in_channels, max(self.kernel_size_list), self.stride,
            groups=self.max_in_channels // self.channels_per_group, bias=bias,
        )
        
        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  # e.g., [3, 5, 7]
        if self.KERNEL_TRANSFORM_MODE is not None:
            # register scaling parameters
            # 7to5_matrix, 5to3_matrix
            scale_params = {}
            for i in range(len(self._ks_set) - 1):
                ks_small = self._ks_set[i]
                ks_larger = self._ks_set[i + 1]
                param_name = '%dto%d' % (ks_larger, ks_small)
                scale_params['%s_matrix' % param_name] = Parameter(torch.eye(ks_small ** 2))
            for name, param in scale_params.items():
                self.register_parameter(name, param)

        self.active_kernel_size = max(self.kernel_size_list)
    
    def get_active_filter(self, in_channel, kernel_size):
        out_channel = in_channel
        max_kernel_size = max(self.kernel_size_list)
        
        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.conv.weight[:out_channel, :in_channel, start:end, start:end]
        if self.KERNEL_TRANSFORM_MODE is not None and kernel_size < max_kernel_size:
            start_filter = self.conv.weight[:out_channel, :in_channel, :, :]  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(_input_filter.size(0), _input_filter.size(1), -1)
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter, self.__getattr__('%dto%d_matrix' % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks ** 2)
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks, target_ks)
                start_filter = _input_filter
            filters = start_filter
        return filters
    
    def forward(self, x, kernel_size=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        in_channel = x.size(1)
        assert in_channel % self.channels_per_group == 0
        
        filters = self.get_active_filter(in_channel, kernel_size).contiguous()
        
        padding = get_same_padding(kernel_size)
        y = F.conv2d(
            x, filters, None, self.stride, padding, self.dilation, in_channel // self.channels_per_group
        )
        return y


class DynamicPointConv2d(nn.Module):
    
    def __init__(self, max_in_channels, max_out_channels, kernel_size=1, stride=1, dilation=1):
        super(DynamicPointConv2d, self).__init__()
        
        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        
        self.conv = nn.Conv2d(
            self.max_in_channels, self.max_out_channels, self.kernel_size, stride=self.stride, bias=False,
        )
        
        self.active_out_channel = self.max_out_channels
    
    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        filters = self.conv.weight[:out_channel, :in_channel, :, :].contiguous()
        
        padding = get_same_padding(self.kernel_size)
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)
        return y

class DynamicPointConv2dWithIdx(nn.Module): # 最终版带有idx的DynamicPointConv2d
    
    def __init__(self, max_in_channels, max_out_channels, kernel_size=1, stride=1, dilation=1):
        super(DynamicPointConv2dWithIdx, self).__init__()
        
        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        
        self.conv = nn.Conv2d(
            self.max_in_channels, self.max_out_channels, self.kernel_size, stride=self.stride, bias=False,
        )
        
        self.active_out_channel = self.max_out_channels
        self.active_in_idx = None
        self.active_out_idx = None
    
    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)

        assert self.active_in_idx or self.active_out_idx
        if self.active_in_idx and self.active_out_idx:
            assert len(self.active_out_idx) == out_channel
            assert len(self.active_in_idx) == in_channel
            filters = self.conv.weight[self.active_out_idx, :, :, :].contiguous()
            filters = filters[:, self.active_in_idx, :, :].contiguous()
        elif self.active_in_idx:
            assert len(self.active_in_idx) == in_channel
            filters = self.conv.weight[:out_channel, self.active_in_idx, :, :].contiguous()
        elif self.active_out_idx:
            assert len(self.active_out_idx) == out_channel
            filters = self.conv.weight[self.active_out_idx, :in_channel, :, :].contiguous()    
        
        padding = get_same_padding(self.kernel_size)
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)
        return y

# class DynamicPointConv2dForAttnOut(nn.Module): # to_q, to_k, to_v时用的conv，由DynamicPointConv2d修改得到
    
#     def __init__(self, max_in_channels, max_qkv_dim, max_num_heads, max_channel_split, kernel_size=1, stride=1, dilation=1): # max_dim可能是q, k或v的dim
#         super(DynamicPointConv2dForAttnOut, self).__init__()
        
#         self.max_in_channels = max_in_channels
#         self.max_qkv_dim = max_qkv_dim
#         self.max_num_heads = max_num_heads
#         self.max_channel_split = max_channel_split # [32, 64, 128, 160]
#         # 通过每部分channel的个数，得到每部分开始的索引
#         self.max_in_channel_start_idx = compute_idx_from_channel_split(self.max_channel_split) # [0, 32, 96, 224]
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.dilation = dilation
        
#         self.conv = nn.Conv2d(
#             self.max_in_channels, self.max_qkv_dim * self.max_num_heads, self.kernel_size, stride=self.stride, bias=False,
#         ) # out_channel为self.max_qkv_dim * self.max_num_heads

#         self.active_qkv_dim = self.max_qkv_dim
#         self.active_num_heads = self.max_num_heads
#         self.active_channel_split = self.max_channel_split

#     def forward(self, x):
#         # 新增     
#         qkv_dim = self.active_qkv_dim
#         num_heads = self.active_num_heads
#         channel_split = self.active_channel_split

#         # in_channel_idx
#         assert len(channel_split) == len(self.max_channel_split)        
#         in_channel_idx = []
#         for i in range(len(self.max_in_channel_start_idx)):
#             for j in range(self.max_in_channel_start_idx[i], self.max_in_channel_start_idx[i]+channel_split[i]):
#                 in_channel_idx.append(j)
#         in_channel = x.size(1)
#         assert len(in_channel_idx) == in_channel

#         # out_channel_idx
#         out_channel_idx = [] # 切片
#         for i in range(num_heads):
#             for j in range(qkv_dim):
#                 out_channel_idx.append(i * self.max_qkv_dim + j)

#         filters = self.conv.weight[out_channel_idx, in_channel_idx, :, :].contiguous()        
#         padding = get_same_padding(self.kernel_size)
#         y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)
#         return y

# class DynamicPointConv2dForAttnIn(nn.Module): # proj时用的conv，由DynamicPointConv2d修改得到
    
#     def __init__(self, max_out_channels, max_qkv_dim, max_num_heads, max_channel_split, kernel_size=1, stride=1, dilation=1): # max_dim可能是q, k或v的dim
#         super(DynamicPointConv2dForAttnIn, self).__init__()
        
#         self.max_out_channels = max_out_channels
#         self.max_qkv_dim = max_qkv_dim
#         self.max_num_heads = max_num_heads
#         self.max_channel_split = max_channel_split # [32, 64, 128, 160]
#         # 通过每部分channel的个数，得到每部分开始的索引
#         self.max_out_channel_start_idx = compute_idx_from_channel_split(self.max_channel_split) # [0, 32, 96, 224]
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.dilation = dilation
        
#         self.conv = nn.Conv2d(
#             self.max_qkv_dim * self.max_num_heads, self.max_out_channels, self.kernel_size, stride=self.stride, bias=False,
#         ) # in_channel为self.max_qkv_dim * self.max_num_heads
        
#         self.active_out_channels = self.max_out_channels
#         self.active_qkv_dim = self.max_qkv_dim
#         self.active_num_heads = self.max_num_heads
#         self.active_channel_split = self.max_channel_split

#     def forward(self, x):
#         # 新增
#         qkv_dim = self.active_qkv_dim
#         num_heads = self.active_num_heads
#         out_channel = self.active_out_channels
#         channel_split = self.active_channel_split

#         # in_channel_idx
#         in_channel = x.size(1)
#         in_channel_idx = [] # 切片
#         for i in range(num_heads):
#             for j in range(qkv_dim):
#                 in_channel_idx.append(i * self.max_qkv_dim + j)
#         assert in_channel == len(in_channel_idx)

#         # out_channel_idx
#         assert len(channel_split) == len(self.max_channel_split)        
#         out_channel_idx = []
#         for i in range(len(self.max_out_channel_start_idx)):
#             for j in range(self.max_out_channel_start_idx[i], self.max_out_channel_start_idx[i]+channel_split[i]):
#                 out_channel_idx.append(j)        
#         assert len(out_channel_idx) == out_channel


#         filters = self.conv.weight[out_channel_idx, in_channel_idx, :, :].contiguous()
        
#         padding = get_same_padding(self.kernel_size)
#         y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)
#         return y

# class DynamicPointConv2dWithOutIdx(nn.Module):
    
#     def __init__(self, max_in_channels, max_out_channels, max_channel_split, kernel_size=1, stride=1, dilation=1):
#         super(DynamicPointConv2dWithOutIdx, self).__init__()
        
#         self.max_in_channels = max_in_channels
#         self.max_out_channels = max_out_channels
#         self.max_channel_split = max_channel_split # [32, 64, 128, 160]
#         # 通过每部分channel的个数，得到每部分开始的索引
#         self.max_out_channel_start_idx = compute_idx_from_channel_split(self.max_channel_split) # [0, 32, 96, 224]
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.dilation = dilation
        
#         self.conv = nn.Conv2d(
#             self.max_in_channels, self.max_out_channels, self.kernel_size, stride=self.stride, bias=False,
#         )
        
#         self.active_out_channel = self.max_out_channels
#         self.active_channel_split = self.max_channel_split
    
#     def forward(self, x, out_channel=None):
#         if out_channel is None:
#             out_channel = self.active_out_channel
#         channel_split = self.active_channel_split

#         assert len(channel_split) == len(self.max_channel_split)        
#         out_channel_idx = []
#         for i in range(len(self.max_out_channel_start_idx)):
#             for j in range(self.max_out_channel_start_idx[i], self.max_out_channel_start_idx[i]+channel_split[i]):
#                 out_channel_idx.append(j)
#         assert len(out_channel_idx) == out_channel

#         in_channel = x.size(1)
#         filters = self.conv.weight[out_channel_idx, :in_channel, :, :].contiguous()
        
#         padding = get_same_padding(self.kernel_size)
#         y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)
#         return y

# class DynamicPointConv2dWithInIdx(nn.Module):
    
#     def __init__(self, max_in_channels, max_out_channels, max_channel_split, kernel_size=1, stride=1, dilation=1):
#         super(DynamicPointConv2dWithInIdx, self).__init__()
        
#         self.max_in_channels = max_in_channels
#         self.max_out_channels = max_out_channels
#         self.max_channel_split = max_channel_split # [32, 64, 128, 160]
#         # 通过每部分channel的个数，得到每部分开始的索引
#         self.max_in_channel_start_idx = compute_idx_from_channel_split(self.max_channel_split) # [0, 32, 96, 224]
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.dilation = dilation
        
#         self.conv = nn.Conv2d(
#             self.max_in_channels, self.max_out_channels, self.kernel_size, stride=self.stride, bias=False,
#         )
        
#         self.active_out_channel = self.max_out_channels
#         self.active_channel_split = self.max_channel_split
    
#     def forward(self, x, out_channel=None):
#         if out_channel is None:
#             out_channel = self.active_out_channel
#         channel_split = self.active_channel_split

#         assert len(channel_split) == len(self.max_channel_split)
#         in_channel_idx = []
#         for i in range(len(self.max_in_channel_start_idx)):
#             for j in range(self.max_in_channel_start_idx[i], self.max_in_channel_start_idx[i]+channel_split[i]):
#                 in_channel_idx.append(j)

#         in_channel = x.size(1)
#         assert in_channel == len(in_channel_idx)
#         filters = self.conv.weight[:out_channel, in_channel_idx, :, :].contiguous()
        
#         padding = get_same_padding(self.kernel_size)
#         y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)
#         return y


class DynamicLinear(nn.Module):
    
    def __init__(self, max_in_features, max_out_features, bias=True):
        super(DynamicLinear, self).__init__()
        
        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.bias = bias
        
        self.linear = nn.Linear(self.max_in_features, self.max_out_features, self.bias)
        
        self.active_out_features = self.max_out_features
    
    def forward(self, x, out_features=None):
        if out_features is None:
            out_features = self.active_out_features
        
        in_features = x.size(1)
        weight = self.linear.weight[:out_features, :in_features].contiguous()
        bias = self.linear.bias[:out_features] if self.bias else None
        y = F.linear(x, weight, bias)
        return y

class DynamicLinearWithIdx(nn.Module):
    
    def __init__(self, max_in_features, max_out_features, bias=True):
        super(DynamicLinearWithIdx, self).__init__()
        
        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.bias = bias
        
        self.linear = nn.Linear(self.max_in_features, self.max_out_features, self.bias)
        
        self.active_out_features = self.max_out_features
        self.active_in_idx = None
        self.active_out_idx = None
    
    def forward(self, x, out_features=None):
        if out_features is None:
            out_features = self.active_out_features
        
        in_features = x.size(1)
        assert self.active_in_idx or self.active_out_idx
        assert in_features == len(self.active_in_idx)

        if self.active_in_idx and self.active_out_idx:
            assert len(self.active_out_idx) == out_features
            assert len(self.active_in_idx) == in_features
            weight = self.linear.weight[self.active_out_idx, self.active_in_idx].contiguous()
            bias = self.linear.bias[self.active_out_idx] if self.bias else None
        elif self.active_in_idx: # 在作为分类头时默认out_features不变，所以走这里
            assert len(self.active_in_idx) == in_features
            weight = self.linear.weight[:out_features, self.active_in_idx].contiguous()
            bias = self.linear.bias[:out_features] if self.bias else None
        elif self.active_out_idx:
            assert len(self.active_out_idx) == out_features
            weight = self.linear.weight[self.active_out_idx, :in_features].contiguous()
            bias = self.linear.bias[self.active_out_idx] if self.bias else None
        
        y = F.linear(x, weight, bias)
        return y

class DynamicSE(SELayer):
    
    def __init__(self, max_channel):
        super(DynamicSE, self).__init__(max_channel)

    def forward(self, x):
        in_channel = x.size(1)
        num_mid = make_divisible(in_channel // self.reduction, divisor=8)

        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        # reduce
        reduce_conv = self.fc.reduce
        reduce_filter = reduce_conv.weight[:num_mid, :in_channel, :, :].contiguous()
        reduce_bias = reduce_conv.bias[:num_mid] if reduce_conv.bias is not None else None
        y = F.conv2d(y, reduce_filter, reduce_bias, 1, 0, 1, 1)
        # relu
        y = self.fc.relu(y)
        # expand
        expand_conv = self.fc.expand
        expand_filter = expand_conv.weight[:in_channel, :num_mid, :, :].contiguous()
        expand_bias = expand_conv.bias[:in_channel] if expand_conv.bias is not None else None
        y = F.conv2d(y, expand_filter, expand_bias, 1, 0, 1, 1)
        # hard sigmoid
        y = self.fc.h_sigmoid(y)

        return x * y

class AllReduce(Function):
    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


class DynamicBatchNorm2d(nn.Module):
    '''
        1. doesn't acculate bn statistics, (momentum=0.)
        2. calculate BN statistics of all subnets after training
        3. bn weights are shared
        https://arxiv.org/abs/1903.05134
        https://detectron2.readthedocs.io/_modules/detectron2/layers/batch_norm.html
    '''
    #SET_RUNNING_STATISTICS = False
    
    def __init__(self, max_feature_dim):
        super(DynamicBatchNorm2d, self).__init__()
        
        self.max_feature_dim = max_feature_dim
        self.bn = nn.BatchNorm2d(self.max_feature_dim)

        #self.exponential_average_factor = 0 #doesn't acculate bn stats
        self.need_sync = False

        # reserved to tracking the performance of the largest and smallest network
        self.bn_tracking = nn.ModuleList(
            [
                nn.BatchNorm2d(self.max_feature_dim, affine=False),
                nn.BatchNorm2d(self.max_feature_dim, affine=False) 
            ]
        )

    def forward(self, x):
        feature_dim = x.size(1)
        if not self.training:
            raise ValueError('DynamicBN only supports training')
        
        bn = self.bn
        # need_sync
        if not self.need_sync:
            return F.batch_norm(
                x, bn.running_mean[:feature_dim], bn.running_var[:feature_dim], bn.weight[:feature_dim],
                bn.bias[:feature_dim], bn.training or not bn.track_running_stats,
                bn.momentum, bn.eps,
            )
        else:
            assert dist.get_world_size() > 1, 'SyncBatchNorm requires >1 world size'
            B, C = x.shape[0], x.shape[1]
            mean = torch.mean(x, dim=[0, 2, 3])
            meansqr = torch.mean(x * x, dim=[0, 2, 3])
            assert B > 0, 'does not support zero batch size'
            vec = torch.cat([mean, meansqr], dim=0)
            vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())
            mean, meansqr = torch.split(vec, C)

            var = meansqr - mean * mean
            invstd = torch.rsqrt(var + bn.eps)
            scale = bn.weight[:feature_dim] * invstd
            bias = bn.bias[:feature_dim] - mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            return x * scale + bias


        #if bn.num_features == feature_dim or DynamicBatchNorm2d.SET_RUNNING_STATISTICS:
        #    return bn(x)
        #else:
        #    exponential_average_factor = 0.0

        #    if bn.training and bn.track_running_stats:
        #        # TODO: if statement only here to tell the jit to skip emitting this when it is None
        #        if bn.num_batches_tracked is not None:
        #            bn.num_batches_tracked += 1
        #            if bn.momentum is None:  # use cumulative moving average
        #                exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
        #            else:  # use exponential moving average
        #                exponential_average_factor = bn.momentum
        #    return F.batch_norm(
        #        x, bn.running_mean[:feature_dim], bn.running_var[:feature_dim], bn.weight[:feature_dim],
        #        bn.bias[:feature_dim], bn.training or not bn.track_running_stats,
        #        exponential_average_factor, bn.eps,
        #    )

class DynamicBatchNorm2dWithIdx(nn.Module):
    '''
        1. doesn't acculate bn statistics, (momentum=0.)
        2. calculate BN statistics of all subnets after training
        3. bn weights are shared
        https://arxiv.org/abs/1903.05134
        https://detectron2.readthedocs.io/_modules/detectron2/layers/batch_norm.html
    '''
    #SET_RUNNING_STATISTICS = False
    
    def __init__(self, max_feature_dim):
        super(DynamicBatchNorm2dWithIdx, self).__init__()
        
        self.max_feature_dim = max_feature_dim
        self.bn = nn.BatchNorm2d(self.max_feature_dim)

        #self.exponential_average_factor = 0 #doesn't acculate bn stats
        self.need_sync = False

        # reserved to tracking the performance of the largest and smallest network
        self.bn_tracking = nn.ModuleList(
            [
                nn.BatchNorm2d(self.max_feature_dim, affine=False),
                nn.BatchNorm2d(self.max_feature_dim, affine=False) 
            ]
        )

        self.active_idx = None

    def forward(self, x):
        feature_dim = x.size(1)
        if not self.training:
            raise ValueError('DynamicBN only supports training')

        assert self.active_idx
        assert len(self.active_idx) == feature_dim

        bn = self.bn
        # need_sync
        if not self.need_sync:
            return F.batch_norm(
                x, bn.running_mean[self.active_idx], bn.running_var[self.active_idx], bn.weight[self.active_idx],
                bn.bias[self.active_idx], bn.training or not bn.track_running_stats,
                bn.momentum, bn.eps,
            )
        else:
            assert dist.get_world_size() > 1, 'SyncBatchNorm requires >1 world size'
            B, C = x.shape[0], x.shape[1]
            mean = torch.mean(x, dim=[0, 2, 3])
            meansqr = torch.mean(x * x, dim=[0, 2, 3])
            assert B > 0, 'does not support zero batch size'
            vec = torch.cat([mean, meansqr], dim=0)
            vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())
            mean, meansqr = torch.split(vec, C)

            var = meansqr - mean * mean
            invstd = torch.rsqrt(var + bn.eps)
            scale = bn.weight[self.active_idx] * invstd
            bias = bn.bias[self.active_idx] - mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            return x * scale + bias

class DynamicBatchNorm1dWithIdx(nn.Module):
    '''
        1. doesn't acculate bn statistics, (momentum=0.)
        2. calculate BN statistics of all subnets after training
        3. bn weights are shared
        https://arxiv.org/abs/1903.05134
        https://detectron2.readthedocs.io/_modules/detectron2/layers/batch_norm.html
    '''
    #SET_RUNNING_STATISTICS = False
    
    def __init__(self, max_feature_dim):
        super(DynamicBatchNorm1dWithIdx, self).__init__()
        
        self.max_feature_dim = max_feature_dim
        self.bn = nn.BatchNorm2d(self.max_feature_dim)

        #self.exponential_average_factor = 0 #doesn't acculate bn stats
        self.need_sync = False

        # reserved to tracking the performance of the largest and smallest network
        self.bn_tracking = nn.ModuleList(
            [
                nn.BatchNorm2d(self.max_feature_dim, affine=False),
                nn.BatchNorm2d(self.max_feature_dim, affine=False) 
            ]
        )

        self.active_idx = None

    def forward(self, x):
        feature_dim = x.size(1)
        if not self.training:
            raise ValueError('DynamicBN only supports training')

        assert self.active_idx
        assert len(self.active_idx) == feature_dim

        bn = self.bn
        # need_sync
        if not self.need_sync:
            return F.batch_norm(
                x, bn.running_mean[self.active_idx], bn.running_var[self.active_idx], bn.weight[self.active_idx],
                bn.bias[self.active_idx], bn.training or not bn.track_running_stats,
                bn.momentum, bn.eps,
            )
        else:
            assert dist.get_world_size() > 1, 'SyncBatchNorm requires >1 world size'
            B, C = x.shape[0], x.shape[1]
            mean = torch.mean(x, dim=[0])
            meansqr = torch.mean(x * x, dim=[0])
            assert B > 0, 'does not support zero batch size'
            vec = torch.cat([mean, meansqr], dim=0)
            vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())
            mean, meansqr = torch.split(vec, C)

            var = meansqr - mean * mean
            invstd = torch.rsqrt(var + bn.eps)
            scale = bn.weight[self.active_idx] * invstd
            bias = bn.bias[self.active_idx] - mean * scale
            scale = scale.reshape(1, -1)
            bias = bias.reshape(1, -1)
            return x * scale + bias

# class DynamicBatchNorm2dForAttn(nn.Module):
#     '''
#         1. doesn't acculate bn statistics, (momentum=0.)
#         2. calculate BN statistics of all subnets after training
#         3. bn weights are shared
#         https://arxiv.org/abs/1903.05134
#         https://detectron2.readthedocs.io/_modules/detectron2/layers/batch_norm.html
#     '''
#     #SET_RUNNING_STATISTICS = False
    
#     def __init__(self, max_qkv_dim, max_num_heads):
#         super(DynamicBatchNorm2dForAttn, self).__init__()
        
#         # self.max_feature_dim = max_feature_dim
#         self.max_qkv_dim = max_qkv_dim
#         self.max_num_heads = max_num_heads

#         # self.bn = nn.BatchNorm2d(self.max_feature_dim)
#         self.bn = nn.BatchNorm2d(self.max_qkv_dim * self.max_num_heads)

#         #self.exponential_average_factor = 0 #doesn't acculate bn stats
#         self.need_sync = False

#         # reserved to tracking the performance of the largest and smallest network
#         self.bn_tracking = nn.ModuleList(
#             [
#                 nn.BatchNorm2d(self.max_qkv_dim * self.max_num_heads, affine=False),
#                 nn.BatchNorm2d(self.max_qkv_dim * self.max_num_heads, affine=False) 
#             ]
#         )

#         self.active_qkv_dim = max_qkv_dim # forward时需要更改这两个属性
#         self.active_num_heads = max_num_heads

#     def forward(self, x):
#         feature_dim = x.size(1)
#         # 新增
#         qkv_dim = self.active_qkv_dim
#         num_heads = self.active_num_heads
#         assert feature_dim == qkv_dim * num_heads
#         idx = [] # 切片
#         for i in range(num_heads):
#             for j in range(qkv_dim):
#                 idx.append(i * self.max_qkv_dim + j)

#         if not self.training:
#             raise ValueError('DynamicBN only supports training')
        
#         bn = self.bn
#         # need_sync
#         if not self.need_sync:
#             # 修改out_channel为切片
#             return F.batch_norm(
#                 x, bn.running_mean[idx], bn.running_var[idx], bn.weight[idx],
#                 bn.bias[idx], bn.training or not bn.track_running_stats,
#                 bn.momentum, bn.eps,
#             )
#         else:
#             assert dist.get_world_size() > 1, 'SyncBatchNorm requires >1 world size'
#             B, C = x.shape[0], x.shape[1]
#             mean = torch.mean(x, dim=[0, 2, 3])
#             meansqr = torch.mean(x * x, dim=[0, 2, 3])
#             assert B > 0, 'does not support zero batch size'
#             vec = torch.cat([mean, meansqr], dim=0)
#             vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())
#             mean, meansqr = torch.split(vec, C)
#             # 修改out_channel为切片
#             var = meansqr - mean * mean
#             invstd = torch.rsqrt(var + bn.eps)
#             scale = bn.weight[idx] * invstd
#             bias = bn.bias[idx] - mean * scale
#             scale = scale.reshape(1, -1, 1, 1)
#             bias = bias.reshape(1, -1, 1, 1)
#             return x * scale + bias

# class DynamicBatchNorm2dWithIdx(nn.Module):
#     '''
#         1. doesn't acculate bn statistics, (momentum=0.)
#         2. calculate BN statistics of all subnets after training
#         3. bn weights are shared
#         https://arxiv.org/abs/1903.05134
#         https://detectron2.readthedocs.io/_modules/detectron2/layers/batch_norm.html
#     '''
#     #SET_RUNNING_STATISTICS = False
    
#     def __init__(self, max_feature_dim, max_feature_split):
#         super(DynamicBatchNorm2dWithIdx, self).__init__()
        
#         self.max_feature_dim = max_feature_dim
#         self.max_feature_split = max_feature_split # [32, 64, 128, 160]
#         # 通过每部分channel的个数，得到每部分开始的索引
#         self.max_feature_start_idx = compute_idx_from_channel_split(self.max_feature_split) # [0, 32, 96, 224]

#         self.bn = nn.BatchNorm2d(self.max_feature_dim)

#         #self.exponential_average_factor = 0 #doesn't acculate bn stats
#         self.need_sync = False

#         # reserved to tracking the performance of the largest and smallest network
#         self.bn_tracking = nn.ModuleList(
#             [
#                 nn.BatchNorm2d(self.max_feature_dim, affine=False),
#                 nn.BatchNorm2d(self.max_feature_dim, affine=False) 
#             ]
#         )

#         self.active_feature_split = max_feature_split

#     def forward(self, x):
#         feature_dim = x.size(1)
#         # 新增
#         feature_split = self.active_feature_split
#         assert len(feature_split) == len(self.max_feature_split)        
#         feature_idx = []
#         for i in range(len(self.max_feature_start_idx)):
#             for j in range(self.max_feature_start_idx[i], self.max_feature_start_idx[i]+feature_split[i]):
#                 feature_idx.append(j)        
#         assert feature_dim == len(feature_idx)

#         if not self.training:
#             raise ValueError('DynamicBN only supports training')
        
#         bn = self.bn
#         # need_sync
#         if not self.need_sync:
#             return F.batch_norm(
#                 x, bn.running_mean[feature_idx], bn.running_var[feature_idx], bn.weight[feature_idx],
#                 bn.bias[feature_idx], bn.training or not bn.track_running_stats,
#                 bn.momentum, bn.eps,
#             )
#         else:
#             assert dist.get_world_size() > 1, 'SyncBatchNorm requires >1 world size'
#             B, C = x.shape[0], x.shape[1]
#             mean = torch.mean(x, dim=[0, 2, 3])
#             meansqr = torch.mean(x * x, dim=[0, 2, 3])
#             assert B > 0, 'does not support zero batch size'
#             vec = torch.cat([mean, meansqr], dim=0)
#             vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())
#             mean, meansqr = torch.split(vec, C)

#             var = meansqr - mean * mean
#             invstd = torch.rsqrt(var + bn.eps)
#             scale = bn.weight[feature_idx] * invstd
#             bias = bn.bias[feature_idx] - mean * scale
#             scale = scale.reshape(1, -1, 1, 1)
#             bias = bias.reshape(1, -1, 1, 1)
#             return x * scale + bias

# class DynamicBatchNorm1dWithIdx(nn.Module): # add for cls head
    
#     def __init__(self, max_feature_dim, max_feature_split):
#         super(DynamicBatchNorm1dWithIdx, self).__init__()
        
#         self.max_feature_dim = max_feature_dim
#         self.max_feature_split = max_feature_split # [32, 64, 128, 160]
#         # 通过每部分channel的个数，得到每部分开始的索引
#         self.max_feature_start_idx = compute_idx_from_channel_split(self.max_feature_split) # [0, 32, 96, 224]

#         self.bn = nn.BatchNorm1d(self.max_feature_dim)

#         #self.exponential_average_factor = 0 #doesn't acculate bn stats
#         self.need_sync = False

#         # reserved to tracking the performance of the largest and smallest network
#         self.bn_tracking = nn.ModuleList(
#             [
#                 nn.BatchNorm1d(self.max_feature_dim, affine=False),
#                 nn.BatchNorm1d(self.max_feature_dim, affine=False) 
#             ]
#         )

#         self.active_feature_split = max_feature_split

#     def forward(self, x):
#         feature_dim = x.size(1)
#         # 新增
#         feature_split = self.active_feature_split
#         assert len(feature_split) == len(self.max_feature_split)        
#         feature_idx = []
#         for i in range(len(self.max_feature_start_idx)):
#             for j in range(self.max_feature_start_idx[i], self.max_feature_start_idx[i]+feature_split[i]):
#                 feature_idx.append(j)        
#         assert feature_dim == len(feature_idx)

#         if not self.training:
#             raise ValueError('DynamicBN only supports training')
        
#         bn = self.bn
#         # need_sync
#         if not self.need_sync:
#             return F.batch_norm(
#                 x, bn.running_mean[:feature_idx], bn.running_var[:feature_idx], bn.weight[:feature_idx],
#                 bn.bias[:feature_idx], bn.training or not bn.track_running_stats,
#                 bn.momentum, bn.eps,
#             )
#         else:
#             assert dist.get_world_size() > 1, 'SyncBatchNorm requires >1 world size'
#             B, C = x.shape[0], x.shape[1]
#             mean = torch.mean(x, dim=[0])
#             meansqr = torch.mean(x * x, dim=[0])
#             assert B > 0, 'does not support zero batch size'
#             vec = torch.cat([mean, meansqr], dim=0)
#             vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())
#             mean, meansqr = torch.split(vec, C)

#             var = meansqr - mean * mean
#             invstd = torch.rsqrt(var + bn.eps)
#             scale = bn.weight[:feature_idx] * invstd
#             bias = bn.bias[:feature_idx] - mean * scale
#             scale = scale.reshape(1, -1)
#             bias = bias.reshape(1, -1)
#             return x * scale + bias

