# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
from .. import builder
from ..builder import SEGMENTORS
# from .base import BaseSegmentor
from mmseg.utils.nn_utils import int2list
from .static_encoder_decoder import StaticEncoderDecoder
from .encoder_decoder import EncoderDecoder
import mmseg.utils.loss_ops as loss_ops

@SEGMENTORS.register_module()
class DynamicEncoderDecoder(EncoderDecoder):
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
                 pretrained=None,
                 init_cfg=None,
                 num_arch_training=4,
                 sandwich_rule=True,
                 distiller=dict(),
                 bn_param=(0., 1e-5),
                 sync_bn=True,
                 ):
        super(EncoderDecoder, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head
        
        # 新增
        self.auxiliary_head = auxiliary_head
        self.pretrained = pretrained
        self.init_cfg = init_cfg
        self.num_arch_training = num_arch_training
        self.sandwich_rule = sandwich_rule
   
        if distiller['name'] == 'cwd':
            self.soft_criterion = loss_ops.ChannelWiseDivergence(tau=distiller['tau'], loss_weight=distiller['loss_weight'])
        elif distiller['name'] == 'kd':
            self.soft_criterion = loss_ops.CriterionPixelWise()
        else:
            self.soft_criterion = None
        self.sync_bn = sync_bn
        
        # self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])
        if getattr(self, 'sync_bn', False):
            self.apply(
                lambda m: setattr(m, 'need_sync', True)) # 对模块的所有children都设置need_sync为True，如果不存在该属性就创建并赋值
        
    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Inplace-distill forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        num_subnet_training = max(2, self.num_arch_training)

        ### compute gradients using sandwich rule ###
        # step 1 sample the largest network, apply regularization to only the largest network
        self.sample_max_subnet()
        self.set_dropout_rate(self.backbone.drop, self.backbone.drop_path_rate) #dropout for supernet
        x = self.extract_feat(img) # 如果没有neck，就是backbone的输出
        losses = dict()
        loss_decode, seg_logits = self._decode_head_forward_train(x, img_metas,
                                                    gt_semantic_seg,
                                                    return_logit=True)
        # loss_decode = {'decode.loss_ce': tensor(3.4209, device='cuda:0', grad_fn=<MulBackward0>), 'decode.acc_seg': tensor([0.0372], device='cuda:0')}
        losses.update(loss_decode)
        # losses = {'decode.loss_ce': tensor(3.4209, device='cuda:0', grad_fn=<MulBackward0>), 'decode.acc_seg': tensor([0.0372], device='cuda:0')}
        # seg_logits.shape = torch.Size([4, 150, 72, 72])
        if self.with_auxiliary_head: 
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
            # losses是个dict，update相当于把loss_aux字段加到dict里
            # losses = {'decode.loss_ce': tensor(3.4209, device='cuda:0', grad_fn=<MulBackward0>), 'decode.acc_seg': tensor([0.0372], device='cuda:0'),
            #           'decode.loss_aux': tensor(3.4209, device='cuda:0', grad_fn=<MulBackward0>)}
        
        with torch.no_grad():
            soft_logits = seg_logits.clone().detach()
        
        #step 2. sample the smallest network and several random networks
        loss_kd_val = 0.
        loss_ce_val = 0.
        sandwich_rule = self.sandwich_rule
        self.set_dropout_rate(0, 0)  #reset dropout rate
        for arch_id in range(1, num_subnet_training):
            if arch_id == num_subnet_training-1 and sandwich_rule:
                self.sample_min_subnet()
            else:
                self.sample_active_subnet()
            
            # calcualting loss
            x = self.extract_feat(img)
            ce_loss, seg_logits = self._decode_head_forward_train(x, img_metas,
                                                        gt_semantic_seg,
                                                        return_logit=True)
            # import pdb; pdb.set_trace()
            if self.soft_criterion:            
                assert seg_logits.shape == soft_logits.shape, 'the shape of output is not equal to soft_logits!'
                loss_kd_val += self.soft_criterion(seg_logits, soft_logits) # shape要一样大
                # 小网络蒸馏loss很小，三个加起来只有tensor(0.0079, device='cuda:0', grad_fn=<AddBackward0>), 之后需要调蒸馏loss权重
            else: # 小网络用ce loss
                loss_ce_val += ce_loss['decode.loss_ce']

        if self.soft_criterion: # 小网络也用蒸馏 loss
            loss_subnet_kd = {'decode.loss_subnet_kd': loss_kd_val}
            losses.update(loss_subnet_kd) # 在BaseSegmentor的_parse_losses方法里，会把losses的key中带'loss'的值加入到最终的loss中
            # losses = {'decode.loss_ce': tensor(3.3932, device='cuda:0', grad_fn=<MulBackward0>), \
            #           'decode.acc_seg': tensor([0.2764], device='cuda:0'), \
            #           'decode.loss_subnet_ce': tensor(0.0079, device='cuda:0', grad_fn=<AddBackward0>)}
        else: # 小网络用ce loss
            loss_subnet_ce = {'decode.loss_subnet_ce': loss_ce_val}
            losses.update(loss_subnet_ce)
            
        return losses

    """ ************** set, sample and get active sub-networks ************** """
    def set_active_subnet(self, width=None, depth=None, kernel_size=None, expand_ratio=None, \
        num_heads=None, key_dim=None, attn_ratio=None, mlp_ratio=None, transformer_depth=None, **kwargs): # bignas用到，用于固定每层的config
        self.backbone.set_active_subnet(width, depth, kernel_size, expand_ratio, \
            num_heads, key_dim, attn_ratio, mlp_ratio, transformer_depth, **kwargs) 

    def get_active_subnet_settings(self): # for eval
        return self.backbone.get_active_subnet_settings() 
    
    def set_dropout_rate(self, dropout=0, drop_path=0):
        self.backbone.set_dropout_rate(dropout, drop_path)
        # head部分的dropout全程都是固定的，因此不用单独写

    """ ************** sample min, max and middle sub-networks ************** """
    def sample_min_subnet(self): # bignas用到
        return self._sample_active_subnet(min_net=True)

    def sample_max_subnet(self): # bignas用到
        return self._sample_active_subnet(max_net=True)

    def sample_active_subnet(self, compute_flops=False): # bignas用到，在bignas中compute_flops=False
        cfg = self._sample_active_subnet(False, False) 
        if compute_flops: # not for bignas
            raise NotImplementedError
            cfg['flops'] = self.compute_active_subnet_flops()
        return cfg

    def _sample_active_subnet(self, min_net=False, max_net=False): # bignas用到
        sample_cfg = lambda candidates, sample_min, sample_max: \
            min(candidates) if sample_min else (max(candidates) if sample_max else random.choice(candidates))

        cfg = {}
        # self.cfg_candidates['expand_ratio'] = [[1], [4, 5, 6], [4, 5, 6], [4, 5, 6], [4, 5, 6], [6], [6]]
        # 表示每个stage的选择，stage内的每个block都一样
        '''tpm + trans'''
        for k in ['width', 'depth', 'kernel_size', 'expand_ratio',\
            'num_heads', 'key_dim', 'attn_ratio', 'mlp_ratio', 'transformer_depth']:
            cfg[k] = []
            for vv in self.backbone.cfg_candidates[k]:
                cfg[k].append(sample_cfg(int2list(vv), min_net, max_net))
        # cfg['width'] = [24, 24, 32, 40, 72, 128, 216, 224, 1984] 每个stage从self.cfg_candidates中选一个，其他同理
        self.set_active_subnet(
            cfg['width'], cfg['depth'], cfg['kernel_size'], cfg['expand_ratio'],
            cfg['num_heads'], cfg['key_dim'], cfg['attn_ratio'], cfg['mlp_ratio'], cfg['transformer_depth']
        )
        
        return cfg

    """ ****************** mutate and crossover ****************** """

    def mutate_and_reset(self, cfg, prob=0.1):
        cfg = copy.deepcopy(cfg)
        pick_another = lambda x, candidates: x if len(candidates) == 1 else random.choice([v for v in candidates if v != x])
        r = random.random()

        # sample channels, depth, kernel_size, expand_ratio
        for k in ['width', 'depth', 'kernel_size', 'expand_ratio',\
            'num_heads', 'key_dim', 'attn_ratio', 'mlp_ratio', 'transformer_depth']:
            for _i, _v in enumerate(cfg[k]):
                r = random.random()
                if r < prob:
                    cfg[k][_i] = pick_another(cfg[k][_i], int2list(self.backbone.cfg_candidates[k][_i]))
        self.set_active_subnet(
            cfg['width'], cfg['depth'], cfg['kernel_size'], cfg['expand_ratio'],
            cfg['num_heads'], cfg['key_dim'], cfg['attn_ratio'], cfg['mlp_ratio'], cfg['transformer_depth']
        )
        return cfg

    def crossover_and_reset(self, cfg1, cfg2, p=0.5):
        def _cross_helper(g1, g2, prob):
            assert type(g1) == type(g2)
            if isinstance(g1, int):
                return g1 if random.random() < prob else g2
            elif isinstance(g1, list):
                return [v1 if random.random() < prob else v2 for v1, v2 in zip(g1, g2)]
            else:
                raise NotImplementedError

        cfg = {}
        for k in ['width', 'depth', 'kernel_size', 'expand_ratio',\
            'num_heads', 'key_dim', 'attn_ratio', 'mlp_ratio', 'transformer_depth']:
            cfg[k] = _cross_helper(cfg1[k], cfg2[k], p)
        self.set_active_subnet(
            cfg['width'], cfg['depth'], cfg['kernel_size'], cfg['expand_ratio'],
            cfg['num_heads'], cfg['key_dim'], cfg['attn_ratio'], cfg['mlp_ratio'], cfg['transformer_depth']
        )
        return cfg

    def get_active_subnet(self, preserve_weight=True):  # 获得静态子网
        with torch.no_grad():
            # 暂时只搜backbone
            backbone = self.backbone.get_active_subnet(preserve_weight)
            # 权重可以传入，最大子网的权重是一样的，之后可以测试下小子网的
            subnet = StaticEncoderDecoder(backbone=backbone,
                                  decode_head=self.decode_head,
                                  neck=self.neck if self.with_neck else None,
                                  auxiliary_head=self.auxiliary_head,
                                  train_cfg=self.train_cfg,
                                  test_cfg=self.test_cfg,
                                  init_cfg=self.init_cfg)
            # self.set_bn_param(**self.get_bn_param())
            return subnet

    # def compute_active_subnet_flops(self):
    #     total_ops = 0
    #     ops, size_out_list = self.backbone.compute_active_subnet_flops()
    #     total_ops += ops
    #     total_ops += self.decode_head.compute_active_subnet_flops(size_out_list)
    #     return total_ops
