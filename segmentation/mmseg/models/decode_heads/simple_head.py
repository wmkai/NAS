# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *


@HEADS.register_module()
class SimpleHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, is_dw=False, **kwargs):
        super(SimpleHead, self).__init__(input_transform='multiple_select', **kwargs)

        embedding_dim = self.channels

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1,
            groups=embedding_dim if is_dw else 1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
    
    def agg_res(self, preds): # 都上采样到最大(1/8)尺寸 然后相加
        outs = preds[0]
        for pred in preds[1:]:
            pred = resize(pred, size=outs.size()[2:], mode='bilinear', align_corners=False)
            outs += pred
        return outs

    def forward(self, inputs): # len(inputs) = 3
        # pdb.set_trace()
        xx = self._transform_inputs(inputs)  # 1/8, 1/16, 1/32 [B, 256, 64, 64] [B, 256, 32, 32] [B, 256, 16, 16]
        x = self.agg_res(xx) # 把xx中所有元素上采样到相同尺寸相加 [B, 256, 64, 64]
        _c = self.linear_fuse(x) # 1x1 conv聚合通道信息，维度不变 [B, 256, 64, 64]
        x = self.cls_seg(_c) # [B, 150, 64, 64] 1/8分辨率
        # for distillation
        # import torch.nn.functional as F
        # x = F.interpolate(x, scale_factor = 2, mode = 'bilinear', align_corners=False) # [B, 150, 128, 128] 1/4分辨率
        return x
    
    # def compute_active_subnet_flops(self, size_out_list):
    #     def count_conv(c_in, c_out, size_out, groups, k):
    #         kernel_ops = k**2
    #         output_elements = c_out * size_out**2
    #         ops = c_in * output_elements * kernel_ops / groups
    #         return ops
        
    #     def count_linear(c_in, c_out):
    #         return c_in * c_out

    #     def count_bn(c_in, size_out, affine):
    #         ops = c_in * size_out * size_out
    #         if affine:
    #             ops *= 2
    #         return ops

    #     def count_act(c_in, size_out):
    #         return c_in * size_out * size_out

    #     total_ops = 0
        
    #     # 1x1 conv + bn + relu
    #     total_ops += count_conv(self.channels, self.channels, size_out_list[0], 1, 1)
    #     total_ops += count_bn(self.channels, size_out_list[0], affine=True)
    #     total_ops += count_act(self.channels, size_out_list[0])
    #     # cls_seg 1x1 conv
    #     total_ops += count_conv(self.channels, self.num_classes, size_out_list[0], 1, 1)
        
    #     return total_ops
