# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
from mmcv.cnn import ConvModule
import pdb
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
            # pdb.set_trace()
            pred = resize(pred, size=outs.size()[2:], mode='bilinear', align_corners=False)
            outs += pred
        return outs

    def forward(self, inputs): # len(inputs) = 3
        # pdb.set_trace()
        xx = self._transform_inputs(inputs)  # 1/8, 1/16, 1/32 [B, 256, 64, 64] [B, 256, 32, 32] [B, 256, 16, 16]
        x = self.agg_res(xx) # 把xx中所有元素上采样到相同尺寸相加 [B, 256, 64, 64]
        _c = self.linear_fuse(x) # 1x1 conv聚合通道信息，维度不变 [B, 256, 64, 64]
        x = self.cls_seg(_c) # [B, 150, 64, 64] 1/8分辨率
        return x