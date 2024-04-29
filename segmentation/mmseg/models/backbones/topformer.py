import math
import torch
from torch import nn
import torch.nn.functional as F
import pdb
from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from mmcv.runner import _load_checkpoint
from mmseg.utils import get_root_logger

from ..builder import BACKBONES


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


class Mlp(nn.Module): # 输入输出维度不变
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0., norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.drop = nn.Dropout(drop)

    def forward(self, x): # [B, 384, 8, 8]
        # pdb.set_trace()
        x = self.fc1(x) # conv实现 [B, 768, 8, 8]
        x = self.dwconv(x) # [B, 768, 8, 8]
        x = self.act(x) # [B, 768, 8, 8]
        x = self.drop(x) # [B, 768, 8, 8]
        x = self.fc2(x) # conv实现 [B, 384, 8, 8]
        x = self.drop(x) # [B, 384, 8, 8]
        return x


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        ks: int,
        stride: int,
        expand_ratio: int,
        activations = None,
        norm_cfg=dict(type='BN', requires_grad=True)
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        assert stride in [1, 2]

        if activations is None:
            activations = nn.ReLU

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
        # pw
            layers.append(Conv2d_BN(inp, hidden_dim, ks=1, norm_cfg=norm_cfg))
            layers.append(activations())
        
        layers.extend([
            # dw
            Conv2d_BN(hidden_dim, hidden_dim, ks=ks, stride=stride, pad=ks//2, groups=hidden_dim, norm_cfg=norm_cfg),
            activations(),
            # pw-linear
            Conv2d_BN(hidden_dim, oup, ks=1, norm_cfg=norm_cfg)
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class TokenPyramidModule(nn.Module):
    def __init__(
        self, 
        cfgs,
        out_indices,
        inp_channel=16,
        activation=nn.ReLU,
        norm_cfg=dict(type='BN', requires_grad=True),
        width_mult=1.):
        super().__init__()
        self.out_indices = out_indices

        self.stem = nn.Sequential(
            Conv2d_BN(3, inp_channel, 3, 2, 1, norm_cfg=norm_cfg),
            activation()
        )
        self.cfgs = cfgs

        self.layers = []
        for i, (k, t, c, s) in enumerate(cfgs):
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = t * inp_channel
            exp_size = _make_divisible(exp_size * width_mult, 8)
            layer_name = 'layer{}'.format(i + 1)
            layer = InvertedResidual(inp_channel, output_channel, ks=k, stride=s, expand_ratio=t, norm_cfg=norm_cfg, activations=activation)
            self.add_module(layer_name, layer)
            inp_channel = output_channel
            self.layers.append(layer_name)

    def forward(self, x):
        outs = []
        x = self.stem(x)
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs # 多尺度特征


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4,
                 activation=None,
                 norm_cfg=dict(type='BN', requires_grad=True),):
        super().__init__() 
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))

    def forward(self, x):  # x (B,N,C) 这个注释有问题，这里的维度应该是B, C, H, W [B, 384, 8, 8]
        # pdb.set_trace()
        B, C, H, W = get_shape(x) 
        # pdb.set_trace()
        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2) # [B, self.num_heads = 8, H * W = 8 x 8, self.key_dim = 16]
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W) # 同qq
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2) # [B, self.num_heads = 8, H * W = 8 x 8, self.d = 32]

        attn = torch.matmul(qq, kk) # [B, 8, 64, 64]
        attn = attn.softmax(dim=-1) # dim = k

        xx = torch.matmul(attn, vv) # [B, 8, 64, 32]

        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W) # [B, 256, 8, 8]
        xx = self.proj(xx) # [B, 384, 8, 8]
        return xx


class Block(nn.Module): # MSA+MLP

    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, activation=act_layer, norm_cfg=norm_cfg) # MSA

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, norm_cfg=norm_cfg) # mlp用卷积实现，两个1x1卷积中间一个dw卷积

    def forward(self, x1): # [B, 384, 8, 8]
        # pdb.set_trace()
        x1 = x1 + self.drop_path(self.attn(x1)) # [B, 384, 8, 8]
        x1 = x1 + self.drop_path(self.mlp(x1)) # [B, 384, 8, 8]
        return x1


class BasicLayer(nn.Module): # 堆叠了transformer block的layer
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                norm_cfg=dict(type='BN2d', requires_grad=True), 
                act_layer=None):
        super().__init__()
        self.block_num = block_num
        idx = 0
        self.transformer_blocks = nn.ModuleList()
        for i in range(len(self.block_num)):
            for j in range(self.block_num[i]):
                self.transformer_blocks.append(Block( # transformer block
                    embedding_dim, key_dim=key_dim[i], num_heads=num_heads[i],
                    mlp_ratio=mlp_ratio[i], attn_ratio=attn_ratio[i],
                    drop=drop, drop_path=drop_path[idx] if isinstance(drop_path, list) else drop_path,
                    norm_cfg=norm_cfg,
                    act_layer=act_layer))
                idx += 1

        # for i in range(self.block_num):
        #     self.transformer_blocks.append(Block( # transformer block
        #         embedding_dim, key_dim=key_dim, num_heads=num_heads,
        #         mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
        #         drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
        #         norm_cfg=norm_cfg,
        #         act_layer=act_layer))

    def forward(self, x):
        # token * N 
        idx = 0
        for i in range(len(self.block_num)):
            for j in range(self.block_num[i]):
                x = self.transformer_blocks[idx](x)
                idx += 1
        return x


class PyramidPoolAgg(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        # pdb.set_trace()
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
        norm_cfg=dict(type='BN', requires_grad=True),
        activations = None,
    ) -> None:
        super(InjectionMultiSum, self).__init__()
        self.norm_cfg = norm_cfg

        self.local_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_act = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
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


class InjectionMultiSumCBR(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        norm_cfg=dict(type='BN', requires_grad=True),
        activations = None,
    ) -> None:
        '''
        local_embedding: conv-bn-relu
        global_embedding: conv-bn-relu
        global_act: conv
        '''
        super(InjectionMultiSumCBR, self).__init__()
        self.norm_cfg = norm_cfg

        self.local_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg)
        self.global_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg)
        self.global_act = ConvModule(inp, oup, kernel_size=1, norm_cfg=None, act_cfg=None)
        self.act = h_sigmoid()

        self.out_channels = oup

    def forward(self, x_l, x_g):
        B, C, H, W = x_l.shape
        local_feat = self.local_embedding(x_l)
        # kernel
        global_act = self.global_act(x_g)
        global_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
        # feat_h
        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)
        out = local_feat * global_act + global_feat
        return out


class FuseBlockSum(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        norm_cfg=dict(type='BN', requires_grad=True),
        activations = None,
    ) -> None:
        super(FuseBlockSum, self).__init__()
        self.norm_cfg = norm_cfg

        if activations is None:
            activations = nn.ReLU

        self.fuse1 = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.fuse2 = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)

        self.out_channels = oup

    def forward(self, x_l, x_h):
        B, C, H, W = x_l.shape
        inp = self.fuse1(x_l)
        kernel = self.fuse2(x_h)
        feat_h = F.interpolate(kernel, size=(H, W), mode='bilinear', align_corners=False)
        out = inp + feat_h
        return out


class FuseBlockMulti(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int = 1,
        norm_cfg=dict(type='BN', requires_grad=True),
        activations = None,
    ) -> None:
        super(FuseBlockMulti, self).__init__()
        self.stride = stride
        self.norm_cfg = norm_cfg
        assert stride in [1, 2]

        if activations is None:
            activations = nn.ReLU

        self.fuse1 = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.fuse2 = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid()

    def forward(self, x_l, x_h):
        B, C, H, W = x_l.shape
        inp = self.fuse1(x_l)
        sig_act = self.fuse2(x_h)
        sig_act = F.interpolate(self.act(sig_act), size=(H, W), mode='bilinear', align_corners=False)
        out = inp * sig_act
        return out


SIM_BLOCK = {
    "fuse_sum": FuseBlockSum,
    "fuse_multi": FuseBlockMulti,

    "muli_sum":InjectionMultiSum,
    "muli_sum_cbr":InjectionMultiSumCBR,
}


@BACKBONES.register_module()
class Topformer(BaseModule):
    def __init__(self,
                 cfgs,
                 stem_channel=16,
                 channels=None,
                 out_channels=None,
                 embed_out_indice=None, 
                 decode_out_indices=[1, 2, 3],
                 depths=[1,1,1,1],
                 key_dim=[16,16,16,16],
                 num_heads=[8,8,8,8],
                 attn_ratios=[2.0,2.0,2.0,2.0],
                 mlp_ratios=[2.0,2.0,2.0,2.0],
                 c2t_stride=2,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_layer=nn.ReLU6,
                 injection_type="muli_sum",
                 init_cfg=None,
                 injection=True):
        super().__init__()
        self.channels = channels
        self.norm_cfg = norm_cfg
        self.injection = injection
        self.embed_dim = sum(channels)
        self.decode_out_indices = decode_out_indices
        self.init_cfg = init_cfg
        if self.init_cfg != None:
            self.pretrained = self.init_cfg['checkpoint']
        self.tpm_time = 0.0 # 用于计时
        self.sase_time = 0.0 # 用于计时
        self.sim_time = 0.0 # 用于计时

        self.tpm = TokenPyramidModule(cfgs=cfgs, out_indices=embed_out_indice, inp_channel=stem_channel, norm_cfg=norm_cfg)
        self.ppa = PyramidPoolAgg(stride=c2t_stride)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.trans = BasicLayer(
            block_num=depths,
            embedding_dim=self.embed_dim,
            key_dim=key_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratios,
            attn_ratio=attn_ratios,
            drop=0, attn_drop=0, 
            drop_path=dpr,
            norm_cfg=norm_cfg,
            act_layer=act_layer)
        
        # SemanticInjectionModule
        self.SIM = nn.ModuleList()
        inj_module = SIM_BLOCK[injection_type] # 默认为"muli_sum"
        if self.injection:
            for i in range(len(channels)):
                if i in decode_out_indices:
                    self.SIM.append(
                        inj_module(channels[i], out_channels[i], norm_cfg=norm_cfg, activations=act_layer))
                else:
                    self.SIM.append(nn.Identity())

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n //= m.groups
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
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
    
    def forward(self, x): # [B, 3, 512, 512]
        # pdb.set_trace()
        # torch.cuda.synchronize()
        # import time; tpm_start = time.time()        
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
            ouputs.append(out)
            return ouputs
        
