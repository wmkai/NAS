# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# adapted from OFA: https://github.com/mit-han-lab/once-for-all

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import BACKBONES
from mmcv.runner import BaseModule

@BACKBONES.register_module()
class HESSStaticSupernet(BaseModule):
    def __init__(self, 
                 stem, 
                 layers, 
                 ppa,
                 transformer_blocks,
                 SIM,
                 cls_head,
                 channels,
                #  out_channels,
                 embed_out_indice,
                 decode_out_indices,#=[1, 2, 3],
                 injection,#=True
                 runtime_depth,
                 trans_runtime_depth
                 ):
        BaseModule.__init__(self)

        self.channels = channels
        self.injection = injection
        # self.embed_dim = sum(channels)
        self.decode_out_indices = decode_out_indices

        self.tpm = StaticTokenPyramidModule(stem=stem, layers=layers, embed_out_indice=embed_out_indice, runtime_depth=runtime_depth)

        # self.ppa = PyramidPoolAgg(stride=c2t_stride)
        self.ppa = ppa

        self.trans = StaticTransformerBasicLayer(transformer_blocks=transformer_blocks, runtime_depth=trans_runtime_depth)

        # SemanticInjectionModule
        self.SIM = SIM

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = cls_head

    def forward(self, x): 

        ouputs = self.tpm(x) # 多尺度特征 list len=4 [B, 32, 128, 128] [B, 64, 64, 64] [B, 128, 32, 32] [B, 160, 16, 16]
        out = self.ppa(ouputs) # 把多尺度特征池化到1/64分辨率，然后在通道维度concat [B, 384, 8, 8], 384 = 32+64+128+160
        out = self.trans(out) # [B, 384, 8, 8] 维度不变
        if self.injection: 
            xx = out.split(self.channels, dim=1) # self.channels = [32, 64, 128, 160], 把out在通道维度上split, 得到len=4的list
            # [B, 32, 8, 8] [B, 64, 8, 8] [B, 128, 8, 8] [B, 160, 8, 8]
            results = []
            for i in range(len(self.channels)):
                if i in self.decode_out_indices: # [1, 2, 3]
                    local_tokens = ouputs[i] # 原始的多尺度特征，尺寸不一样
                    global_semantics = xx[i] # 经过transformer聚合空间信息后的多尺度特征，尺寸是一样的
                    out_ = self.SIM[i](local_tokens, global_semantics) # 4个SIM块
                    results.append(out_)
                    # list len=3,[B, 256, 64, 64] [B, 256, 32, 32] [B, 256, 16, 16]
            return results
        else:
            ouputs.append(out)# return ouputs
            x = self.avg_pool(ouputs[-1]).squeeze(-1).squeeze(-1)
            # if self.active_dropout_rate:
            #     x = F.dropout(x, p=float(self.active_dropout_rate), training=self.training)
            x = self.cls_head(x)
            return x

    def weight_initialization(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)): # 可能还有其他norm
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def reset_running_stats_for_calibration(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.SyncBatchNorm): # 可能还有其他norm
                m.training = True
                m.momentum = None # cumulative moving average
                m.reset_running_stats()


class StaticTokenPyramidModule(nn.Module):
    def __init__(
        self, 
        stem,
        layers,
        embed_out_indice,
        runtime_depth
        ):
        super().__init__()        
        self.stem = stem
        self.layers = nn.ModuleList(layers) # 传入时是个list，要包装起来
        self.embed_out_indice = embed_out_indice # 表示要哪些stage id的输出
        self.runtime_depth = runtime_depth # [2, 3, 3, 3, 4] = len(self.layers)

    def forward(self, x):
        outs = []
        x = self.stem(x)
        
        sum_val, return_block_id = 0, [] # 返回的block的id [5, 8, 11, 15]
        for i in range(len(self.runtime_depth)):
            sum_val += self.runtime_depth[i]
            if i in self.embed_out_indice:
                return_block_id.append(sum_val)

        # for i, layer_name in enumerate(self.layers):
        for i, layer in enumerate(self.layers):
            # layer = getattr(self, layer_name)
            x = layer(x)
            if i + 1 in return_block_id: # i从0开始，所以要+1
                outs.append(x)
        return outs # 多尺度特征
        
class StaticTransformerBasicLayer(nn.Module):
    def __init__(
        self, 
        transformer_blocks,
        runtime_depth # 这里的runtime_depth没有用到，因为传入的transformer_blocks已经是削减过的了
        ):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(transformer_blocks)
    
    def forward(self, x):
        for layer in self.transformer_blocks:
            x = layer(x)
        return x