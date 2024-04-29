# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import copy
import mmcv
import torch
from mmcv.cnn import get_model_complexity_info
from mmcv.cnn.utils import revert_sync_batchnorm
from mmseg.models import build_segmentor
from mmseg.utils import get_device, setup_multi_processes

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test flops')
    parser.add_argument('config', help='test config file path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))    
    cfg.device = get_device()   

    # 测给定子网的flops
    subnet_cfg = {'width': [16, 16, 24, 48, 96, 128],
                  'depth': [1, 2, 2, 2, 3], 
                  'kernel_size': [3, 3, 5, 3, 5], 
                  'expand_ratio': [1, 4, 3, 3, 6], 
                  'num_heads': [6, 6, 6, 6], 
                  'key_dim': [16, 16, 16, 16], 
                  'attn_ratio': [2.0, 2.0, 2.0, 2.0], 
                  'mlp_ratio': [2.0, 2.0, 2.0, 2.0], 
                  'transformer_depth': [1, 1, 1, 1]}

    model.set_active_subnet( # dynamic_model
        subnet_cfg['width'], subnet_cfg['depth'], subnet_cfg['kernel_size'],
        subnet_cfg['expand_ratio'], subnet_cfg['num_heads'], subnet_cfg['key_dim'], 
        subnet_cfg['attn_ratio'], subnet_cfg['mlp_ratio'], subnet_cfg['transformer_depth']
    )
    # 测最大或最小子网的flops
    # model.sample_max_subnet()
    # model.sample_min_subnet()
    subnet = model.get_active_subnet().cuda()  
    subnet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(subnet)
    flops, params = compute_flops_and_params(subnet, print_per_layer_stat=True)
    print('Subnet flops: {} params: {}'.format(flops, params))

def compute_flops_and_params(model, print_per_layer_stat=False): # 单卡
    tmp_model = copy.deepcopy(model)
    tmp_model.forward = tmp_model.forward_dummy        
    tmp_model.eval()            
    flops, params = get_model_complexity_info(tmp_model, (3, 512, 512), print_per_layer_stat=print_per_layer_stat)
    flops = float(flops.split(' ')[0])
    return flops, params

if __name__ == '__main__':
    main()