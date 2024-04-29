# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# copy from NASVIT
import math
from turtle import forward
from typing import OrderedDict
import torch
import torch.nn as nn

from .static_layers import ConvBnActLayer, StaticAttention, StaticMLP
from .dynamic_ops import DynamicPointConv2dWithIdx, DynamicSeparableConv2d, DynamicBatchNorm2d, DynamicBatchNorm2dWithIdx
from .nn_utils import int2list, get_net_device, copy_bn, copy_bn_idx, build_activation, compute_idx_from_channel_split, make_divisible

class DynamicMlp(nn.Module): # drop根据当前active的通道数计算，以及dw中的bias以及linear1的参数，需要修改
    def __init__(self, mlp_ratio_list, in_dim_list, out_dim_list, 
            max_channel_split: list, bias=True, act_func=None, drop=0.
        ): 
        super(DynamicMlp, self).__init__()
        # self.hidden_features_list = int2list(hidden_features_list)
        self.mlp_ratio_list = int2list(mlp_ratio_list)
        self.in_dim_list = int2list(in_dim_list)
        self.out_dim_list = int2list(out_dim_list)
        self.max_channel_split = max_channel_split
        # 通过每部分channel的个数，得到每部分开始的索引
        self.max_channel_start_idx = compute_idx_from_channel_split(self.max_channel_split)
        self.bias = bias

        max_hidden_features = make_divisible(round(max(self.in_dim_list) * max(self.mlp_ratio_list)))

        self.fc1 = nn.Sequential(OrderedDict([
                ('conv', DynamicPointConv2dWithIdx(max(self.in_dim_list), max_hidden_features, 1, 1, 1)),
                ('bn', DynamicBatchNorm2d(max_feature_dim=max_hidden_features))
            ]))
        self.dwconv = DynamicSeparableConv2d(max_in_channels=max_hidden_features, kernel_size_list=[3], stride=1, dilation=1, channels_per_group=1, bias=self.bias)
        self.act = build_activation(act_func, inplace=True)
        self.fc2 = nn.Sequential(OrderedDict([
                ('conv', DynamicPointConv2dWithIdx(max_hidden_features, max(self.out_dim_list), 1, 1, 1)),
                ('bn', DynamicBatchNorm2dWithIdx(max_feature_dim=max(self.out_dim_list)))
            ]))
        self.drop = nn.Dropout(drop)
        # self.active_hidden_features = max(self.hidden_features_list)
        self.active_mlp_ratio = max(self.mlp_ratio_list)
        self.active_out_dim = max(self.out_dim_list)
        # 新增
        self.active_channel_split = max_channel_split

    def forward(self, x):
        # split_channel_idx
        assert len(self.active_channel_split) == len(self.max_channel_split)        
        split_channel_idx = []
        for i in range(len(self.max_channel_start_idx)):
            for j in range(self.max_channel_start_idx[i], self.max_channel_start_idx[i]+self.active_channel_split[i]):
                split_channel_idx.append(j)

        # self.fc1.conv.active_out_channel = self.active_hidden_features
        in_dim = x.size(1)
        self.fc1.conv.active_out_channel = \
            make_divisible(round(in_dim * self.active_mlp_ratio), 8)
        self.fc1.conv.active_in_idx = split_channel_idx
        self.fc2.conv.active_out_channel = self.active_out_dim
        self.fc2.conv.active_out_idx = split_channel_idx
        self.fc2.bn.active_idx = split_channel_idx

        x = self.fc1(x) # conv实现 [B, 768, 8, 8]
        x = self.dwconv(x) # [B, 768, 8, 8]
        x = self.act(x) # [B, 768, 8, 8]
        x = self.drop(x) # [B, 768, 8, 8]
        x = self.fc2(x) # conv实现 [B, 384, 8, 8]
        x = self.drop(x) # [B, 384, 8, 8]
        return x
    
    def get_active_subnet(self, in_dim, preserve_weight=True): 
        # MLP层的输入维度和attn层的输出维度相同，因此输入权重只要取到active_out_dim
        hidden_features =  make_divisible(round(in_dim * self.active_mlp_ratio), 8)
        sub_layer = StaticMLP(fc1=ConvBnActLayer(in_dim, hidden_features, 1, 1, 1, groups=1, bias=False, use_bn=True, act_func=None), 
                              dwconv=ConvBnActLayer(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features, bias=self.bias, use_bn=False, act_func=None),  # dw没有bn
                              act=self.act,
                              fc2=ConvBnActLayer(hidden_features, self.active_out_dim, 1, 1, 1, groups=1, bias=False, use_bn=True, act_func=None), 
                              drop=self.drop)
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer
        
        assert len(self.active_channel_split) == len(self.max_channel_split)        
        split_channel_idx = []
        for i in range(len(self.max_channel_start_idx)):
            for j in range(self.max_channel_start_idx[i], self.max_channel_start_idx[i]+self.active_channel_split[i]):
                split_channel_idx.append(j)
        assert in_dim == len(split_channel_idx) == self.active_out_dim
        
        sub_layer.fc1.conv.weight.data.copy_(self.fc1.conv.conv.weight.data[:hidden_features, split_channel_idx, :, :])
        copy_bn(sub_layer.fc1.bn, self.fc1.bn.bn)
        sub_layer.dwconv.conv.weight.data.copy_(self.dwconv.conv.weight.data[:hidden_features, :, :, :]) # dw卷积第二维为1
        if self.bias:
            sub_layer.dwconv.conv.bias.data.copy_(self.dwconv.conv.bias.data[:hidden_features])
        sub_layer.fc2.conv.weight.data.copy_(self.fc2.conv.conv.weight.data[split_channel_idx, :hidden_features, :, :])
        copy_bn_idx(sub_layer.fc2.bn, self.fc2.bn.bn, split_channel_idx)
        return sub_layer

class DynamicAttention(nn.Module):
    def __init__(self, in_dim_list, out_dim_list, key_dim_list, attn_ratio_list, num_heads_list, 
            max_channel_split: list, act_func=None
        ):
        super(DynamicAttention, self).__init__()
        
        self.in_dim_list = int2list(in_dim_list)
        self.out_dim_list = int2list(out_dim_list)
        self.key_dim_list = int2list(key_dim_list)  
        # self.value_dim_list = int2list(value_dim_list)   
        self.attn_ratio_list = int2list(attn_ratio_list)   
        self.num_heads_list = int2list(num_heads_list)
        self.max_channel_split = max_channel_split
        # 通过每部分channel的个数，得到每部分开始的索引
        self.max_channel_start_idx = compute_idx_from_channel_split(self.max_channel_split)
        
        self.act = build_activation(act_func, inplace=True)
        self.scale = max(self.key_dim_list) ** -0.5 # 没用到scale        

        self.to_q = nn.Sequential(OrderedDict([
                ('conv', DynamicPointConv2dWithIdx(max_in_channels=max(self.in_dim_list), max_out_channels=max(self.key_dim_list) * max(self.num_heads_list))),
                ('bn', DynamicBatchNorm2dWithIdx(max_feature_dim=max(self.key_dim_list) * max(self.num_heads_list)))
            ]))
        self.to_k = nn.Sequential(OrderedDict([
                ('conv', DynamicPointConv2dWithIdx(max_in_channels=max(self.in_dim_list), max_out_channels=max(self.key_dim_list) * max(self.num_heads_list))),
                ('bn', DynamicBatchNorm2dWithIdx(max_feature_dim=max(self.key_dim_list) * max(self.num_heads_list)))
            ]))
        self.to_v = nn.Sequential(OrderedDict([
                ('conv', DynamicPointConv2dWithIdx(max_in_channels=max(self.in_dim_list), max_out_channels=int(max(self.key_dim_list) * max(self.attn_ratio_list)) * max(self.num_heads_list))),
                ('bn', DynamicBatchNorm2dWithIdx(max_feature_dim=int(max(self.key_dim_list) * max(self.attn_ratio_list)) * max(self.num_heads_list)))
            ]))
        self.proj = nn.Sequential(OrderedDict([
                ('act', self.act),
                ('conv', DynamicPointConv2dWithIdx(max_in_channels=int(max(self.key_dim_list) * max(self.attn_ratio_list)) * max(self.num_heads_list), max_out_channels=max(self.out_dim_list))),
                ('bn', DynamicBatchNorm2dWithIdx(max_feature_dim=max(self.out_dim_list))) 
            ]))

        self.active_out_dim = max(self.out_dim_list)
        self.active_key_dim = max(self.key_dim_list) # q,k dim
        self.active_attn_ratio = max(self.attn_ratio_list) # qk_dim x attn_ratio = v_dim
        self.active_num_heads = max(self.num_heads_list)
        self.active_channel_split = max_channel_split
    
    def forward(self, x): # forward时对子模块active属性的修改，参考dynamic_layers.py

        # assert len(self.active_channel_split) == len(self.max_channel_split)        
        # split_channel_idx = []
        # for i in range(len(self.max_channel_start_idx)):
        #     for j in range(self.max_channel_start_idx[i], self.max_channel_start_idx[i]+self.active_channel_split[i]):
        #         split_channel_idx.append(j)

        # q_channel_idx, k_channel_idx, v_channel_idx = [], [], []
        # for i in range(self.active_num_heads):
        #     for j in range(self.active_key_dim):
        #         q_channel_idx.append(i * max(self.key_dim_list) + j)
        #         k_channel_idx.append(i * max(self.key_dim_list) + j)
        # for i in range(self.active_num_heads):
        #     # for j in range(self.active_value_dim):
        #     for j in range(int(self.active_key_dim * self.active_attn_ratio)):            
        #         # v_channel_idx.append(i * max(self.value_dim_list) + j)
        #         v_channel_idx.append(i * int(max(self.key_dim_list) * max(self.attn_ratio_list)) + j)

        idx_qk, idx_v, idx_dim = [], [], [] # 切片，用于取对应索引的参数
        for i in range(self.active_num_heads):
            for j in range(self.active_key_dim):
                idx_qk.append(i * max(self.key_dim_list) + j)
        for i in range(self.active_num_heads):
            # for j in range(self.active_value_dim):
            for j in range(int(self.active_key_dim * self.active_attn_ratio)):            
                # idx_v.append(i * max(self.value_dim_list) + j)
                idx_v.append(i * int(max(self.key_dim_list) * max(self.attn_ratio_list)) + j) 

        for i in range(len(self.max_channel_start_idx)):
            for j in range(self.max_channel_start_idx[i], self.max_channel_start_idx[i]+self.active_channel_split[i]):
                idx_dim.append(j)       
        
        # to_q, to_k, to_v
        self.to_q.conv.active_in_idx, self.to_k.conv.active_in_idx, self.to_v.conv.active_in_idx = \
            idx_dim, idx_dim, idx_dim
        self.to_q.conv.active_out_idx, self.to_k.conv.active_out_idx, self.to_v.conv.active_out_idx = \
            idx_qk, idx_qk, idx_v
        self.to_q.conv.active_out_channel, self.to_k.conv.active_out_channel, self.to_v.conv.active_out_channel = \
            self.active_key_dim * self.active_num_heads, self.active_key_dim * self.active_num_heads, int(self.active_key_dim * self.active_attn_ratio) * self.active_num_heads
        self.to_q.bn.active_idx, self.to_k.bn.active_idx, self.to_v.bn.active_idx = \
            idx_qk, idx_qk, idx_v

        # proj   
        self.proj.conv.active_in_idx, self.proj.conv.active_out_idx, self.proj.conv.active_out_channel = idx_v, idx_dim, self.active_out_dim
        self.proj.bn.active_idx = idx_dim

        B, C, H, W = x.shape
        qq = self.to_q(x).reshape(B, self.active_num_heads, self.active_key_dim, H * W).permute(0, 1, 3, 2) # [B, self.num_heads = 8, H * W = 8 x 8, self.key_dim = 16]
        kk = self.to_k(x).reshape(B, self.active_num_heads, self.active_key_dim, H * W) # 同qq
        vv = self.to_v(x).reshape(B, self.active_num_heads, int(self.active_key_dim * self.active_attn_ratio), H * W).permute(0, 1, 3, 2) # [B, self.num_heads = 8, H * W = 8 x 8, self.d = 32]

        attn = torch.matmul(qq, kk) # [B, 8, 64, 64]
        attn = attn.softmax(dim=-1) # dim = k
        xx = torch.matmul(attn, vv) # [B, 8, 64, 32]        
        xx = xx.permute(0, 1, 3, 2).reshape(B, int(self.active_key_dim * self.active_attn_ratio) * self.active_num_heads, H, W) # [B, 256, 8, 8]
        xx = self.proj(xx) # [B, 384, 8, 8]
        return xx
    

    def get_active_subnet(self, in_dim, preserve_weight=True):
        to_q = ConvBnActLayer(in_dim, self.active_key_dim * self.active_num_heads, 1, 1, 1, 1, False, True, None) # bias=False, use_bn=True, act_func=None
        to_k = ConvBnActLayer(in_dim, self.active_key_dim * self.active_num_heads, 1, 1, 1, 1, False, True, None)
        to_v = ConvBnActLayer(in_dim, int(self.active_key_dim * self.active_attn_ratio) * self.active_num_heads, 1, 1, 1, 1, False, True, None)
        proj = nn.Sequential(OrderedDict([
                ('act', self.act),
                ('conv', ConvBnActLayer(int(self.active_key_dim * self.active_attn_ratio) * self.active_num_heads, self.active_out_dim, 1, 1, 1, 1, False, True, None)),
            ]))
        # sub_layer = StaticAttention(to_q, to_k, to_v, proj, self.active_key_dim, self.active_value_dim, self.active_num_heads)
        sub_layer = StaticAttention(to_q, to_k, to_v, proj, self.active_key_dim, self.active_num_heads, self.active_attn_ratio)
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer
        
        # 新增
        idx_qk, idx_v, idx_dim = [], [], [] # 切片，用于取对应索引的参数
        for i in range(self.active_num_heads):
            for j in range(self.active_key_dim):
                idx_qk.append(i * max(self.key_dim_list) + j)
        for i in range(self.active_num_heads):
            # for j in range(self.active_value_dim):
            for j in range(int(self.active_key_dim * self.active_attn_ratio)):            
                # idx_v.append(i * max(self.value_dim_list) + j)
                idx_v.append(i * int(max(self.key_dim_list) * max(self.attn_ratio_list)) + j)     
        for i in range(len(self.max_channel_start_idx)):
            for j in range(self.max_channel_start_idx[i], self.max_channel_start_idx[i]+self.active_channel_split[i]):
                idx_dim.append(j)

        temp_to_q_weight = self.to_q.conv.conv.weight.data[idx_qk, :, :, :]
        temp_to_k_weight = self.to_k.conv.conv.weight.data[idx_qk, :, :, :]
        temp_to_v_weight = self.to_v.conv.conv.weight.data[idx_v, :, :, :]
        sub_layer.to_q.conv.weight.data.copy_(temp_to_q_weight[:, idx_dim, :, :])  # out, in
        sub_layer.to_k.conv.weight.data.copy_(temp_to_k_weight[:, idx_dim, :, :])
        sub_layer.to_v.conv.weight.data.copy_(temp_to_v_weight[:, idx_dim, :, :])
        copy_bn_idx(sub_layer.to_q.bn, self.to_q.bn.bn, idx_qk)
        copy_bn_idx(sub_layer.to_k.bn, self.to_k.bn.bn, idx_qk)
        copy_bn_idx(sub_layer.to_v.bn, self.to_v.bn.bn, idx_v)

        temp_proj_weight = self.proj.conv.conv.weight[idx_dim, :, :, :]
        sub_layer.proj.conv.conv.weight.data.copy_(temp_proj_weight[:, idx_v, :, :])
        copy_bn_idx(sub_layer.proj.conv.bn, self.proj.bn.bn, idx_dim)
        # 这些模块都没有bias
        return sub_layer

