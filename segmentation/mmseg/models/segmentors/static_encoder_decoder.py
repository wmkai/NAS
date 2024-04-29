# Copyright (c) OpenMMLab. All rights reserved.
from base64 import decode
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class StaticEncoderDecoder(EncoderDecoder):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(EncoderDecoder, self).__init__(init_cfg)

        # self.backbone = builder.build_backbone(backbone)
        self.backbone = backbone
        if neck is not None:
            # self.neck = builder.build_neck(neck)
            self.neck = neck
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        # self.decode_head = builder.build_head(decode_head)
        self.decode_head = decode_head
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            # if isinstance(auxiliary_head, list):
            #     self.auxiliary_head = nn.ModuleList()
            #     for head_cfg in auxiliary_head:
            #         self.auxiliary_head.append(builder.build_head(head_cfg))
            # else:
            #     self.auxiliary_head = builder.build_head(auxiliary_head)
            self.auxiliary_head = auxiliary_head

    def reset_running_stats_for_calibration(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.SyncBatchNorm): # 可能还有其他norm
                m.training = True
                m.momentum = None # cumulative moving average
                m.reset_running_stats()
    

   
