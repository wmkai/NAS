import enum
import math
import torch
from torch import nn
import torch.nn.functional as F
import random
import copy
import numpy as np
# from collections import OrderedDict
# from mmcv.cnn import ConvModule
# from mmcv.cnn import build_norm_layer
# from mmcv.runner import BaseModule

from mmseg.utils.dynamic_layers import DynamicMBConvLayer, DynamicConvBnActLayer, \
    DynamicShortcutLayer, DynamicInjectionMultiSum, DynamicClsHead
from mmseg.utils.static_layers import MobileInvertedResidualBlock, PyramidPoolAgg, DropPath, \
TransformerBlock 
from mmseg.utils.transformer import DynamicMlp, DynamicAttention
from mmseg.utils.nn_utils import int2list, make_divisible

from mmcv.runner import BaseModule
from mmcv.runner import _load_checkpoint
from mmseg.utils import get_root_logger
from .hess_static_supernet import HESSStaticSupernet
from ..builder import BACKBONES
 
@BACKBONES.register_module()
class HESSDynamicSupernet(BaseModule):
    def __init__(self,
                 supernet=None,
                 fix_backbone=False,
                 fix_trans=False,
                 in_chans=3):
        BaseModule.__init__(self)        
        '''**************** start supernet config ****************''' 
        self.supernet = supernet # supernet是yml中的supernet_config
        # embed_out_indice表示选哪几个mb block的输出拼接起来作为transformer的输入，
        self.embed_out_indice = self.supernet.embed_out_indice     
        self.c2t_stride = self.supernet.c2t_stride
        self.drop = self.supernet.drop
        self.drop_path_rate = self.supernet.drop_path_rate
        self.pretrained = self.supernet.pretrained

        # self.channels = self.supernet.SIM.channels # for SIM, in_channel
        self.out_channels = self.supernet.SIM.out_channels # for SIM, out_channel
        self.injection = self.supernet.SIM.injection # for SIM
         # decode_out_indices表示从embed_out_indice中选哪几个作为SIM的输入
        self.decode_out_indices = self.supernet.SIM.decode_out_indices # for SIM
        # self.injection_type = self.supernet.SIM.injection_type # for SIM

        # self.embed_dim = sum(self.channels)
        self.active_dropout_rate = 0  
        # self.drop_path_only_last_two_stages = self.supernet.drop_path_only_last_two_stages
        '''******************* end supernet config *******************''' 
        # tpm
        self.stage_names = ['stem', 'mb1', 'mb2', 'mb3', 'mb4', 'mb5']
        self.width_list, self.depth_list, self.ks_list, self.expand_ratio_list = [], [], [], []
        for name in self.stage_names:
            block_cfg = getattr(self.supernet, name)
            self.width_list.append(block_cfg.c)
            if name.startswith('mb'): # mb block独有深度d，ks，expand_ratio t
                self.depth_list.append(block_cfg.d)
                self.ks_list.append(block_cfg.k)
                self.expand_ratio_list.append(block_cfg.t)
         
        self.dynamic_channel_split = []
        for i, name in enumerate(self.stage_names[1:]):
            if i in self.supernet.embed_out_indice:
                block_cfg = getattr(self.supernet, name)
                self.dynamic_channel_split.append(max(block_cfg.c))
        # trans
        self.trans_stage_names = ['trans1', 'trans2', 'trans3', 'trans4']
        self.num_heads_list, self.key_dim_list, self.attn_ratio_list, \
            self.mlp_ratio_list, self.transformer_depth_list = [], [], [], [], []
        for name in self.trans_stage_names:
            trans_cfg = getattr(self.supernet, name)
            # self.out_dim_list.append(trans_cfg.out_dim)
            self.num_heads_list.append(trans_cfg.num_heads)
            self.key_dim_list.append(trans_cfg.key_dim)
            self.attn_ratio_list.append(trans_cfg.attn_ratio)
            self.mlp_ratio_list.append(trans_cfg.mlp_ratio)
            self.transformer_depth_list.append(trans_cfg.d)

        self.cfg_candidates = {
            # tpm
            'width': self.width_list, # [[1,2,3],[1,2,3],[1,2,3]]
            'depth': self.depth_list, # [[1,2,3],[1,2,3],[1,2,3]]
            'kernel_size': self.ks_list, # [[1,2,3],[1,2,3],[1,2,3]]
            'expand_ratio': self.expand_ratio_list, # [[1,2,3],[1,2,3],[1,2,3]]
            # trans
            # 'out_dim': self.out_dim_list, # [[1,2,3],[1,2,3],[1,2,3]]
            'num_heads': self.num_heads_list, # [[1,2,3],[1,2,3],[1,2,3]]
            'key_dim': self.key_dim_list, # [[1,2,3],[1,2,3],[1,2,3]]
            'attn_ratio': self.attn_ratio_list, # [[1,2,3],[1,2,3],[1,2,3]]
            'mlp_ratio': self.mlp_ratio_list, # [[1,2,3],[1,2,3],[1,2,3]]
            'transformer_depth': self.transformer_depth_list, # [1,2,3]
        }
        assert (not fix_backbone) or (not fix_trans) # 最多固定一个部分
        if fix_backbone: # 缩小backbone搜索空间
            for k in ['width', 'depth', 'kernel_size', 'expand_ratio']:
                for i in range(len(self.cfg_candidates[k])):
                    self.cfg_candidates[k][i] = int2list(max(self.cfg_candidates[k][i]))
        elif fix_trans: # 缩小trans搜索空间
            for k in ['num_heads', 'key_dim', 'attn_ratio', 'mlp_ratio', 'transformer_depth']: # transformer部分采样最大值
                for i in range(len(self.cfg_candidates[k])):
                    self.cfg_candidates[k][i] = int2list(max(self.cfg_candidates[k][i]))

        self.tpm = TokenPyramidModule(supernet=self.supernet, 
                                      in_chans=in_chans,
                                      stage_names=self.stage_names[1:], # 这里的stage_names只包含mb块的 
                                      embed_out_indice=self.embed_out_indice, 
                                      )
        self.ppa = PyramidPoolAgg(stride=self.c2t_stride)
        self.trans = TransformerBasicLayer(
            supernet=self.supernet,
            trans_stage_names=self.trans_stage_names,
            # embedding_dim=self.embed_dim,
            embedding_dim=sum(self.dynamic_channel_split),
            channel_split=self.dynamic_channel_split,
            drop=0., 
            drop_path=0., # 0.1，传入后会为每层单独计算一个dpr
        )        
        # SemanticInjectionModule
        self.SIM = nn.ModuleList()
        # inj_module = SIM_BLOCK[self.supernet.SIM.injection_type] # 默认为"muli_sum"
        inj_module = DynamicInjectionMultiSum
        if self.supernet.SIM.injection:
            for i in range(len(self.supernet.embed_out_indice)):
                if i in self.supernet.SIM.decode_out_indices:
                    self.SIM.append( # SIM模块的激活函数默认用h_sigmoid，不受传入参数的影响
                        inj_module(self.dynamic_channel_split[i], self.supernet.SIM.out_channels[i],\
                             norm_cfg=self.supernet.SIM.norm_cfg))
                else:
                    self.SIM.append(nn.Identity())
        
        # set bn param在dynamic_encoder_decoder中执行了

        # imagenet classifier
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = DynamicClsHead(in_feature_list=sum(self.dynamic_channel_split), max_channel_split=self.dynamic_channel_split, out_features=1000)


    def init_weights(self): # 在mmseg框架中会自动调用
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n //= m.groups
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) and m.affine == True: 
                # 其中有一些bn用于跟踪最大和最小网络的性能，这些bn的affine=False，没有weight，这些bn的name中带有'bn_tracking'
                m.weight.data.fill_(1)
                # dict(self.named_modules())['tpm.stem.bn.bn_tracking.0'].weight
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            checkpoint = _load_checkpoint(self.pretrained, logger=logger, map_location='cpu')
            if 'state_dict_ema' in checkpoint:
                state_dict = checkpoint['state_dict_ema']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            self.load_state_dict(state_dict, False)            
            logger.info('Finished loading backbone weights from pretrained model!')
            
    
    def forward(self, x): # [B, 3, 512, 512]
        ouputs = self.tpm(x) # 多尺度特征 list len=4 [B, 32, 128, 128] [B, 64, 64, 64] [B, 128, 32, 32] [B, 160, 16, 16]
        out = self.ppa(ouputs) # 把多尺度特征池化到1/64分辨率，然后在通道维度concat [B, 384, 8, 8], 384 = 32+64+128+160
        out = self.trans(out) # [B, 384, 8, 8] 维度不变

        if self.supernet.SIM.injection:
            # xx = out.split(self.channels, dim=1) # self.channels = [32, 64, 128, 160], 把out在通道维度上split, 得到len=4的list
            xx = out.split(self.dynamic_channel_split, dim=1)
            # xx = [B, 32, 8, 8] [B, 64, 8, 8] [B, 128, 8, 8] [B, 160, 8, 8]
            results = []
            # for i in range(len(self.supernet.SIM.channels)):
            for i in range(len(self.supernet.embed_out_indice)):
                if i in self.supernet.SIM.decode_out_indices: # [1, 2, 3]
                    local_tokens = ouputs[i] # 原始的多尺度特征，尺寸不一样
                    global_semantics = xx[i] # 经过transformer聚合空间信息后的多尺度特征，尺寸是一样的
                    out_ = self.SIM[i](local_tokens, global_semantics) # 4个SIM块
                    results.append(out_)
                    # list len=3,[B, 256, 64, 64] [B, 256, 32, 32] [B, 256, 16, 16]
            return results
        else: # for classification
            ouputs.append(out)
            # return ouputs
            x = self.avg_pool(ouputs[-1]).squeeze(-1).squeeze(-1)
            if self.active_dropout_rate:
                x = F.dropout(x, p=float(self.active_dropout_rate), training=self.training)
            x = self.cls_head(x)
            return x

    """ set, sample and get active sub-networks """
    def set_active_subnet(self, width=None, depth=None, kernel_size=None, expand_ratio=None, \
         num_heads=None, key_dim=None, attn_ratio=None, mlp_ratio=None, transformer_depth=None, **kwargs): # bignas用到，用于固定每层的config
        # 该函数用于sample出cfg后，把每层的配置按照cfg中的设置好
        assert len(depth) == len(kernel_size) == len(expand_ratio) == len(width) - 1 
        '''tpm'''
        # set channel split
        self.dynamic_channel_split = []
        # first conv
        self.tpm.stem.active_out_channel = width[0] 
        in_channel = width[0] 
        # width长度比其他几项多2，包括first_conv和last_conv的width
        for stage_id, (c, k, e, d) in enumerate(zip(width[1:], kernel_size, expand_ratio, depth)): # stage
            start_idx, end_idx = min(self.tpm.block_group_info[stage_id]), max(self.tpm.block_group_info[stage_id])
            for layer_id in range(start_idx, start_idx+d): # stage内每个layer都一样
                layer = self.tpm.layers[layer_id] # layer是MobileInvertedResidualBlock，
                #layer output channels
                layer.mobile_inverted_conv.active_out_channel = c # block.mobile_inverted_conv是DynamicMBConvLayer
                # if layer.shortcut is not None:
                #     layer.shortcut.active_out_channel = c
                if in_channel == c and layer.mobile_inverted_conv.stride == 1:
                    layer.use_shortcut = True
                else:
                    layer.use_shortcut = False
                #dw kernel size
                layer.mobile_inverted_conv.active_kernel_size = k
                #dw expansion ration
                layer.mobile_inverted_conv.active_expand_ratio = e
                in_channel = c
            if stage_id in self.embed_out_indice:
                self.dynamic_channel_split.append(c)
        #IRBlocks repated times
        for i, d in enumerate(depth):
            self.tpm.runtime_depth[i] = min(len(self.tpm.block_group_info[i]), d) # 每个stage运行时的深度 [2, 5, 6, 6, 8, 8, 2]
        '''trans'''      
        # 根据当前层数来设置transformer中每层的drop_path_rate
        # dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.trans.runtime_depth))]
        # idx = 0
        for stage_id, (nh, kd, ar, mr, d) in enumerate(zip(num_heads, key_dim, attn_ratio, mlp_ratio, transformer_depth)):
            start_idx, end_idx = min(self.trans.block_group_info[stage_id]), max(self.trans.block_group_info[stage_id])
            for layer_id in range(start_idx, start_idx+d):
                layer = self.trans.transformer_blocks[layer_id]
                # layer.attn.active_out_dim = od
                layer.attn.active_out_dim = sum(self.dynamic_channel_split)
                layer.attn.active_key_dim = kd
                layer.attn.active_attn_ratio = ar
                layer.attn.active_num_heads = nh
                layer.mlp.active_out_dim = sum(self.dynamic_channel_split)
                layer.mlp.active_mlp_ratio = mr
                # 新增
                layer.attn.active_channel_split = self.dynamic_channel_split
                layer.mlp.active_channel_split = self.dynamic_channel_split
                # if isinstance(layer.drop_path, DropPath): # 当dpr==0时,layer.drop_path为nn.Identity()
                #     layer.drop_path.drop_prob = dpr[idx]
                # idx += 1
        for i, d in enumerate(transformer_depth):
            self.trans.runtime_depth[i] = min(len(self.trans.block_group_info[i]), d)
        
        # 新增
        self.cls_head.active_channel_split = self.dynamic_channel_split

    def get_active_subnet_settings(self): # for eval
        width, depth, kernel_size, expand_ratio= [], [], [], []
        out_dim, num_heads, key_dim, attn_ratio, mlp_ratio, transformer_depth = [], [], [], [], [], []
        '''tpm'''
        #first conv
        width.append(self.tpm.stem.active_out_channel)
        for stage_id in range(len(self.tpm.block_group_info)):
            start_idx = min(self.tpm.block_group_info[stage_id])
            # 每个stage内第一个block的config就是该stage内所有block的config
            layer = self.tpm.layers[start_idx]  #first block
            width.append(layer.mobile_inverted_conv.active_out_channel)
            kernel_size.append(layer.mobile_inverted_conv.active_kernel_size)
            expand_ratio.append(layer.mobile_inverted_conv.active_expand_ratio)
            depth.append(self.tpm.runtime_depth[stage_id])
        '''trans'''
        for stage_id in range(len(self.trans.block_group_info)):
            start_idx = min(self.trans.block_group_info[stage_id])
            layer = self.trans.transformer_blocks[start_idx]
            out_dim.append(layer.attn.active_out_dim)
            num_heads.append(layer.attn.active_num_heads)
            key_dim.append(layer.attn.active_key_dim)
            attn_ratio.append(layer.attn.active_attn_ratio)
            mlp_ratio.append(layer.mlp.active_mlp_ratio)
            transformer_depth.append(self.trans.runtime_depth[stage_id])
        channel_split = layer.attn.active_channel_split

        return {
            # tpm
            'width': width,
            'kernel_size': kernel_size,
            'expand_ratio': expand_ratio,
            'depth': depth,
            # trans
            'out_dim': out_dim,
            'num_heads': num_heads,
            'key_dim': key_dim,
            'attn_ratio': attn_ratio,
            'mlp_ratio': mlp_ratio,
            'transformer_depth': transformer_depth,
            'channel_split': channel_split
        }

    def set_dropout_rate(self, drop=0, drop_path=0):
        self.active_dropout_rate = drop # 分类时 transformer最后一层的输出经过dropout
        for i in range(len(self.trans.transformer_blocks)):
            self.trans.transformer_blocks[i].drop_path.drop_prob = drop_path * float(i) / len(self.trans.transformer_blocks)

    # sample_min_subnet, sample_max_subnet, sample_active_subnet调用的都是_sample_active_subnet，只是传的参数不同
    def sample_min_subnet(self): # bignas用到
        return self._sample_active_subnet(min_net=True)

    def sample_max_subnet(self): # bignas用到
        return self._sample_active_subnet(max_net=True)
    
    # def sample_active_subnet(self, compute_flops=False): # bignas用到，在bignas中compute_flops=False
    #     cfg = self._sample_active_subnet(
    #         False, False
    #     ) 
    #     if compute_flops: # not for bignas
    #         raise NotImplementedError
    #         cfg['flops'], _ = self.compute_active_subnet_flops()
    #     return cfg

    def _sample_active_subnet(self, min_net=False, max_net=False): # bignas用到
        sample_cfg = lambda candidates, sample_min, sample_max: \
            min(candidates) if sample_min else (max(candidates) if sample_max else random.choice(candidates))

        cfg = {}
        # self.cfg_candidates['expand_ratio'] = [[1], [4, 5, 6], [4, 5, 6], [4, 5, 6], [4, 5, 6], [6], [6]]
        # 表示每个stage的选择，stage内的每个block都一样
        for k in ['width', 'depth', 'kernel_size', 'expand_ratio',\
            'num_heads', 'key_dim', 'attn_ratio', 'mlp_ratio', 'transformer_depth']:
            cfg[k] = []
            for vv in self.cfg_candidates[k]:
                cfg[k].append(sample_cfg(int2list(vv), min_net, max_net))
        # cfg['width'] = [24, 24, 32, 40, 72, 128, 216, 224, 1984] 每个stage从self.cfg_candidates中选一个，其他同理
        self.set_active_subnet(
            cfg['width'], cfg['depth'], cfg['kernel_size'], cfg['expand_ratio'],
            cfg['num_heads'], cfg['key_dim'], cfg['attn_ratio'], cfg['mlp_ratio'], cfg['transformer_depth']
        )        
        return cfg

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
                    cfg[k][_i] = pick_another(cfg[k][_i], int2list(self.cfg_candidates[k][_i]))
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

    def get_active_subnet(self, preserve_weight=True): 
        with torch.no_grad():
            '''tpm'''
            stem = self.tpm.stem.get_active_subnet(3, preserve_weight)
            layers = []
            input_channel = stem.out_channels
            # tpm blocks
            for stage_id, layer_idx in enumerate(self.tpm.block_group_info):
                depth = self.tpm.runtime_depth[stage_id]
                active_idx = layer_idx[:depth]
                stage_layers = []
                for idx in active_idx:
                    stage_layers.append(MobileInvertedResidualBlock(
                        self.tpm.layers[idx].mobile_inverted_conv.get_active_subnet(input_channel, preserve_weight),
                        # self.tpm.layers[idx].shortcut.get_active_subnet(input_channel, preserve_weight) if self.tpm.layers[idx].shortcut is not None else None
                    ))
                    stage_layers[-1].use_shortcut = self.tpm.layers[idx].use_shortcut
                    input_channel = stage_layers[-1].mobile_inverted_conv.out_channels 
                layers += stage_layers
            '''trans'''
            transformer_blocks = []
            # input_channel = self.embed_dim
            input_channel = sum(self.dynamic_channel_split)
            # transformer blocks
            for stage_id, layer_idx in enumerate(self.trans.block_group_info):
                depth = self.trans.runtime_depth[stage_id]
                active_idx = layer_idx[:depth]
                stage_layers = []
                for idx in active_idx:
                    stage_layers.append(TransformerBlock(
                        attn=self.trans.transformer_blocks[idx].attn.get_active_subnet(
                            in_dim=input_channel, preserve_weight=preserve_weight),
                        mlp=self.trans.transformer_blocks[idx].mlp.get_active_subnet(
                            in_dim=self.trans.transformer_blocks[idx].attn.active_out_dim, # mlp的输入dim为attn块的输出dim
                            preserve_weight=preserve_weight),
                        drop_path=self.trans.transformer_blocks[idx].drop_path.drop_prob
                    ))
                    input_channel = stage_layers[-1].mlp.fc2.out_channels
                transformer_blocks += stage_layers
            # SIM blocks
            SIM_blocks = nn.ModuleList()
            for i, inj_module in enumerate(self.SIM):
                if type(inj_module) == nn.Identity:
                    SIM_blocks.append(inj_module)
                else:
                    SIM_blocks.append(inj_module.get_active_subnet(
                        inp=self.dynamic_channel_split[i]
                    ))
            # cls_head
            cls_head = self.cls_head.get_active_subnet(in_features=sum(self.dynamic_channel_split))

            _subnet = HESSStaticSupernet(stem=stem,  
                                      layers=layers, # 传入后会用ModuleList包装
                                      ppa=self.ppa,
                                      transformer_blocks=transformer_blocks, # 传入后会用ModuleList包装
                                      SIM=SIM_blocks,
                                      cls_head=cls_head,
                                      channels=self.dynamic_channel_split,
                                      embed_out_indice=self.embed_out_indice,
                                      decode_out_indices=self.decode_out_indices,
                                      injection=self.injection,
                                      runtime_depth=self.tpm.runtime_depth,
                                      trans_runtime_depth=self.trans.runtime_depth
            )
            # set bn param在dynamic_encoder_decoder中执行了
            return _subnet

class TokenPyramidModule(nn.Module):
    def __init__(
        self, 
        supernet=None, # 新增
        in_chans=3,
        stage_names=None, # 新增
        embed_out_indice=None,
        # bn_param=(0., 1e-5), # 新增
        ):
        super().__init__()
        self.supernet = supernet 

        self.stage_names = stage_names
        self.embed_out_indice = embed_out_indice

        #first conv layer, including conv, bn, act
        out_channel_list, act_func, stride = \
            self.supernet.stem.c, self.supernet.stem.act_func, self.supernet.stem.s
        self.stem = DynamicConvBnActLayer(
            in_channel_list=int2list(in_chans), out_channel_list=out_channel_list, 
            kernel_size=3, stride=stride, act_func=act_func,
        )
        
        # inverted residual blocks
        self.block_group_info = [] 
        # [[0, 1], [2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13, 14]],不包括stem
        layers = []
        _block_index = 0
        feature_dim = out_channel_list
        for stage_id, key in enumerate(self.stage_names): #不包括stem
            block_cfg = getattr(self.supernet, key) # yml里每个stage的信息
            width = block_cfg.c # [16, 24]
            n_block = max(block_cfg.d) # max(1, 2) = 2
            act_func = block_cfg.act_func # 'swish'
            ks = block_cfg.k # [3, 5]
            expand_ratio_list = block_cfg.t # [1]
            use_se = block_cfg.se # False
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            # [[0, 1], [2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13, 14]],不包括stem
            _block_index += n_block

            output_channel = width # [16, 24]
            for i in range(n_block):
                stride = block_cfg.s if i == 0 else 1 # 每个stage里只有第一个block的stride有可能是2
                if min(expand_ratio_list) >= 4:
                    expand_ratio_list = [_s for _s in expand_ratio_list if _s >= 4] if i == 0 else expand_ratio_list
                mobile_inverted_conv = DynamicMBConvLayer( # 包括pw+dw+pw，还有可能包含se，传到MobileInvertedResidualBlock里
                    in_channel_list=feature_dim, 
                    out_channel_list=output_channel, 
                    kernel_size_list=ks,
                    expand_ratio_list=expand_ratio_list, 
                    stride=stride, 
                    act_func=act_func, 
                    use_se=use_se,
                    channels_per_group=getattr(self.supernet, 'channels_per_group', 1)
                )
                # shortcut = DynamicShortcutLayer(feature_dim, output_channel, reduction=stride)
                # layers.append(MobileInvertedResidualBlock(mobile_inverted_conv, shortcut))
                layers.append(MobileInvertedResidualBlock(mobile_inverted_conv))
                feature_dim = output_channel
        self.layers = nn.ModuleList(layers) # len(self.layers) = 37 x
        # runtime_depth
        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info] # [2, 5, 6, 6, 8, 8, 2] 每个stage最多有多少block
                 
    
    def forward(self, x):
        outs = []
        x = self.stem(x)
        # layers
        for stage_id, layer_idx in enumerate(self.block_group_info):
        # [[0, 1], [2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13, 14]],不包括first_conv和last_conv
            depth = self.runtime_depth[stage_id]
            active_idx = layer_idx[:depth]
            for idx in active_idx:
                x = self.layers[idx](x)
            if stage_id in self.embed_out_indice: # stage内block的数量不确定，因此使用每个stage的最终输出作为多尺度特征
                outs.append(x)
        return outs # 多尺度特征


class TransformerBasicLayer(nn.Module):
    def __init__(
        self, 
        supernet=None, # 新增
        trans_stage_names=None,
        embedding_dim=None,
        channel_split=None,
        drop=0., 
        drop_path=0.,
        ):
        super().__init__()
        self.supernet = supernet
        self.trans_stage_names=trans_stage_names
        self.block_group_info = []
        transformer_blocks = []
        _block_index = 0
        feature_dim = int2list(embedding_dim) # 表示transformer输入的dim数，这里int转为list
        # 用于计算每层对应的drop_path_rate
        max_depth = 0
        for stage_id, key in enumerate(self.trans_stage_names):
            block_cfg = getattr(self.supernet, key)
            max_depth += max(block_cfg.d)
        # dpr = [x.item() for x in torch.linspace(0, drop_path, max_depth)]  
        # idx = 0 # 用于取对应层的dpr
        for stage_id, key in enumerate(self.trans_stage_names):
            block_cfg = getattr(self.supernet, key)
            # out_dim = block_cfg.out_dim
            out_dim = int2list(embedding_dim) # sum(self.dynamic_channel_split)
            num_heads = block_cfg.num_heads
            key_dim = block_cfg.key_dim
            attn_ratio = block_cfg.attn_ratio
            mlp_ratio = block_cfg.mlp_ratio
            n_block = max(block_cfg.d)
            act_func = block_cfg.act_func
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            # [[0, 1], [0, 1], [0, 1], [0, 1]]
            _block_index += n_block
            # out_dim_list = out_dim
            for i in range(n_block):
                attn = DynamicAttention(in_dim_list=feature_dim, 
                                        out_dim_list=out_dim, 
                                        key_dim_list=key_dim, 
                                        attn_ratio_list=attn_ratio, 
                                        num_heads_list=num_heads,
                                        max_channel_split=channel_split, 
                                        act_func=act_func)
                mlp = DynamicMlp(mlp_ratio_list=mlp_ratio,  
                                 in_dim_list=out_dim, 
                                 out_dim_list=out_dim, 
                                 max_channel_split=channel_split, 
                                 bias=True, 
                                 act_func=act_func, 
                                 drop=drop)
                transformer_blocks.append(TransformerBlock(
                    attn=attn, mlp=mlp, drop_path=drop_path # 0.
                ))
                feature_dim = out_dim
                # idx += 1
        self.transformer_blocks = nn.ModuleList(transformer_blocks)
        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

    
    def forward(self, x):
        for stage_id, layer_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = layer_idx[:depth]
            for idx in active_idx:
                x = self.transformer_blocks[idx](x)
        return x


    





