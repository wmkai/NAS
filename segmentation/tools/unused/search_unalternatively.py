# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import time
from unittest import result
import warnings

import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg import digit_version
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import build_ddp, build_dp, get_device, setup_multi_processes

import random
import copy
import numpy as np
import torch.distributed as dist
from mmseg.utils.nn_utils import int2list
from mmcv.cnn import get_model_complexity_info

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint-a', help='checkpoint file of model_a')
    # parser.add_argument('checkpoint-b', help='checkpoint file of model_b')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    # parser.add_argument('--constraint_flops', type=float, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    # assert args.constraint_flops != 0
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    assert cfg.constraint_flops != 0
    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True


    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        cfg.gpu_ids = [args.gpu_id]
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(f'The gpu-ids is reset from {cfg.gpu_ids} to '
                          f'{cfg.gpu_ids[0:1]} to avoid potential error in '
                          'non-distribute testing time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
    elif rank == 0:
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
        mmcv.mkdir_or_exist(osp.abspath(work_dir))      

    seed = init_random_seed(None, 'cuda') # 让不同gpu上采样的模型是相同的，否则
    set_random_seed(seed, deterministic=False)

    '''*************** val loader ***************'''
    # build the val_dataloader
    val_dataset = build_dataset(cfg.data.val)
    # The default loader config
    val_loader_cfg = dict(
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        shuffle=False)
    # The overall dataloader settings
    val_loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader','test_dataloader'
        ]
    })
    val_loader_cfg = {
        **val_loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('val_dataloader', {})
    }
    val_loader = build_dataloader(val_dataset, **val_loader_cfg)

    '''*************** train loader ***************'''
    # build the val_dataloader
    train_dataset = build_dataset(cfg.data.train)
    # The default loader config
    train_loader_cfg = dict(
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=seed,
        drop_last=True)
    # The overall dataloader settings
    train_loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader','test_dataloader'
        ]
    })
    train_loader_cfg = {**train_loader_cfg, **cfg.data.get('train_dataloader', {})}
    train_loader = build_dataloader(train_dataset, **train_loader_cfg)

    seed = init_random_seed(None, 'cuda') # 让不同gpu上采样的模型是相同的，否则
    set_random_seed(seed, deterministic=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    cfg.model.backbone.fix_backbone, cfg.model.backbone.fix_trans = True, False
    model_a = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    cfg.model.backbone.fix_backbone, cfg.model.backbone.fix_trans = False, True
    model_b = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    checkpoint_a = load_checkpoint(model_a, cfg.fix_backbone_ckpt, map_location='cpu')
    checkpoint_b = load_checkpoint(model_b, cfg.fix_trans_ckpt, map_location='cpu')
    # model_a
    if 'CLASSES' in checkpoint_a.get('meta', {}):
        model_a.CLASSES = checkpoint_a['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model_a.CLASSES = val_dataset.CLASSES
    if 'PALETTE' in checkpoint_a.get('meta', {}):
        model_a.PALETTE = checkpoint_a['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model_a.PALETTE = val_dataset.PALETTE
    # model_b
    if 'CLASSES' in checkpoint_b.get('meta', {}):
        model_b.CLASSES = checkpoint_b['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model_b.CLASSES = val_dataset.CLASSES
    if 'PALETTE' in checkpoint_b.get('meta', {}):
        model_b.PALETTE = checkpoint_b['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model_b.PALETTE = val_dataset.PALETTE
    
    eval_kwargs = {} if args.eval_options is None else args.eval_options
    eval_on_format_results = (
        args.eval is not None and 'cityscapes' in args.eval)
    if eval_on_format_results:
        assert len(args.eval) == 1, 'eval on format results is not ' \
                                    'applicable for metrics other than ' \
                                    'cityscapes'
    if args.format_only or eval_on_format_results:
        if 'imgfile_prefix' in eval_kwargs:
            tmpdir = eval_kwargs['imgfile_prefix']
        else:
            tmpdir = '.format_cityscapes'
            eval_kwargs.setdefault('imgfile_prefix', tmpdir)
        mmcv.mkdir_or_exist(tmpdir)
    else:
        tmpdir = None

    cfg.device = get_device()
    rank, _ = get_dist_info()
    
    supernet = cfg.model.backbone.supernet
    cfg_candidates = build_cfg_candidates(supernet)

    # supernet A
    benchmarks_a = []
    for i in range(300):
        subnet_cfg = model_a.sample_active_subnet()
        subnet = model_a.get_active_subnet()
        # compute flops
        subnet_for_flops = copy.deepcopy(subnet)
        subnet_for_flops.forward = subnet_for_flops.forward_dummy
        subnet_for_flops = build_dp_or_ddp(subnet_for_flops, distributed, cfg)          
        subnet_for_flops.eval()            
        flops, params = get_model_complexity_info(subnet_for_flops, (3, 512, 512), print_per_layer_stat=False)
        flops = float(flops.split(' ')[0])
        # eval
        subnet = build_dp_or_ddp(subnet, distributed, cfg)
        bn_calibration(subnet, train_loader, cfg.post_bn_calibration_batch_num)
        results = validate_subnet(subnet, args, distributed, val_loader,
            eval_kwargs, eval_on_format_results)

        if args.eval:
            eval_kwargs.update(metric=args.eval)
            metric = val_dataset.evaluate(results, **eval_kwargs)
            mIoU = metric['mIoU']      
            benchmarks_a.append({
                'subnet_cfg': subnet_cfg,
                'flops': flops,
                'mIoU': mIoU
            })
            print('Fix backbone iteration:{} flops:{} mIoU:{}'.format(i, flops, mIoU))
    pareto_subnet_info_a = list(filter(
        lambda x: is_pareto(x, benchmarks_a), benchmarks_a))
    sorted(pareto_subnet_info_a, key=lambda d: d['mIoU'], reverse=True)

    
    # supernet B
    benchmarks_b = []
    for i in range(300):
        subnet_cfg = model_b.sample_active_subnet()
        subnet = model_b.get_active_subnet()
        # compute flops
        subnet_for_flops = copy.deepcopy(subnet)
        subnet_for_flops.forward = subnet_for_flops.forward_dummy
        subnet_for_flops = build_dp_or_ddp(subnet_for_flops, distributed, cfg)          
        subnet_for_flops.eval()            
        flops, params = get_model_complexity_info(subnet_for_flops, (3, 512, 512), print_per_layer_stat=False)
        flops = float(flops.split(' ')[0])
        # eval
        subnet = build_dp_or_ddp(subnet, distributed, cfg)
        bn_calibration(subnet, train_loader, cfg.post_bn_calibration_batch_num)
        results = validate_subnet(subnet, args, distributed, val_loader,
            eval_kwargs, eval_on_format_results)
        if args.eval:
            eval_kwargs.update(metric=args.eval)
            metric = val_dataset.evaluate(results, **eval_kwargs)
            mIoU = metric['mIoU']
            benchmarks_b.append({
                'subnet_cfg': subnet_cfg,
                'flops': flops,
                'mIoU': mIoU
            })
            print('Fix transformer iteration:{} flops:{} mIoU:{}'.format(i, flops, mIoU))
    pareto_subnet_info_b = list(filter(
        lambda x: is_pareto(x, benchmarks_b), benchmarks_b))
    sorted(pareto_subnet_info_b, key=lambda d: d['mIoU'], reverse=True)

    print('pareto_subnet_info_a:',pareto_subnet_info_a)
    print('pareto_subnet_info_b:',pareto_subnet_info_b)

    merged_benchmarks = []
    for i, a in enumerate(pareto_subnet_info_a):
        for j, b in enumerate(pareto_subnet_info_b):
            a_subnet_cfg, b_subnet_cfg = a['subnet_cfg'], b['subnet_cfg']
            merged_subnet_cfg = {
                'width': b_subnet_cfg['width'],
                'depth': b_subnet_cfg['depth'],
                'kernel_size': b_subnet_cfg['kernel_size'],
                'expand_ratio': b_subnet_cfg['expand_ratio'],
                'num_heads': a_subnet_cfg['num_heads'],
                'key_dim': a_subnet_cfg['key_dim'],
                'attn_ratio': a_subnet_cfg['attn_ratio'],
                'mlp_ratio': a_subnet_cfg['mlp_ratio'],
                'transformer_depth': a_subnet_cfg['transformer_depth'],
            }
            miou_rank = i + j
            model_a.set_active_subnet( # only for computing flops
                merged_subnet_cfg['width'], merged_subnet_cfg['depth'], merged_subnet_cfg['kernel_size'],
                merged_subnet_cfg['expand_ratio'], merged_subnet_cfg['num_heads'], merged_subnet_cfg['key_dim'], 
                merged_subnet_cfg['attn_ratio'], merged_subnet_cfg['mlp_ratio'], merged_subnet_cfg['transformer_depth']
            )
            subnet_for_flops = model_a.get_active_subnet()
            # compute flops
            subnet_for_flops.forward = subnet_for_flops.forward_dummy
            subnet_for_flops = build_dp_or_ddp(subnet_for_flops, distributed, cfg)          
            subnet_for_flops.eval()            
            flops, params = get_model_complexity_info(subnet_for_flops, (3, 512, 512), print_per_layer_stat=False)
            flops = float(flops.split(' ')[0])
            print('flops:{} miou_rank:{}'.format(flops, miou_rank))
            merged_benchmarks.append({
                'subnet_cfg': merged_subnet_cfg,
                'flops': flops,
                'miou_rank': miou_rank
            })
            del subnet_for_flops, flops, params
    
    merged_subnet_info_list = list(filter(
    lambda k: k['flops'] < cfg.constraint_flops + 0.02,
    sorted(merged_benchmarks, key=lambda d: d['miou_rank'])))  # 从小到大
    assert len(merged_subnet_info_list) != 0
    best_merged_subnet_info = merged_subnet_info_list[0]
    best_merged_subnet_cfg = best_merged_subnet_info['subnet_cfg']
    best_merged_subnet_mIoU_rank_metric = best_merged_subnet_info['miou_rank']   
    best_merged_subnet_flops = best_merged_subnet_info['flops']  
    print("Best Architecture:\n{}\n mIoU rank:{} flops: {}".format(
        best_merged_subnet_cfg, best_merged_subnet_mIoU_rank_metric, best_merged_subnet_flops))

def bn_calibration(model, train_loader, post_bn_calibration_batch_num):
    model.eval()
    with torch.no_grad():
        model.module.reset_running_stats_for_calibration()
        for batch_idx, x in enumerate(train_loader):
            img = x['img'].data[0].cuda()
            img_metas = x['img_metas'].data[0]
            gt_semantic_seg = x['gt_semantic_seg'].data[0].cuda()
            if batch_idx >= post_bn_calibration_batch_num:
                break
            model(img=img, img_metas=img_metas, return_loss=True, gt_semantic_seg=gt_semantic_seg)  #forward only            


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()

# 这个函数就是给各个随即模块设置随机种子，在之前的./tools/train.py中设置随机种子那里有调用
def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def build_cfg_candidates(supernet):
    # tpm
    stage_names = ['stem', 'mb1', 'mb2', 'mb3', 'mb4', 'mb5']
    width_list, depth_list, ks_list, expand_ratio_list = [], [], [], []
    for name in stage_names:
        block_cfg = getattr(supernet, name)
        width_list.append(block_cfg.c)
        if name.startswith('mb'): # mb block独有深度d，ks，expand_ratio t
            depth_list.append(block_cfg.d)
            ks_list.append(block_cfg.k)
            expand_ratio_list.append(block_cfg.t)
    # trans
    trans_stage_names = ['trans1', 'trans2', 'trans3', 'trans4']
    num_heads_list, key_dim_list, attn_ratio_list, \
        mlp_ratio_list, transformer_depth_list = [], [], [], [], []
    for name in trans_stage_names:
        trans_cfg = getattr(supernet, name)
        num_heads_list.append(trans_cfg.num_heads)
        key_dim_list.append(trans_cfg.key_dim)
        attn_ratio_list.append(trans_cfg.attn_ratio)
        mlp_ratio_list.append(trans_cfg.mlp_ratio)
        transformer_depth_list.append(trans_cfg.d)
    cfg_candidates = {
        # tpm
        'width': width_list, 
        'depth': depth_list, 
        'kernel_size': ks_list, 
        'expand_ratio': expand_ratio_list, 
        # trans
        'num_heads': num_heads_list, 
        'key_dim': key_dim_list, 
        'attn_ratio': attn_ratio_list, 
        'mlp_ratio': mlp_ratio_list, 
        'transformer_depth': transformer_depth_list,
    }
    return cfg_candidates

def build_dp_or_ddp(subnet, distributed, cfg):
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        subnet = revert_sync_batchnorm(subnet)
        subnet = build_dp(subnet, cfg.device, device_ids=cfg.gpu_ids)
    else:
        subnet = build_ddp(
            subnet, cfg.device, device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)
    return subnet

def bn_calibration(model, train_loader, post_bn_calibration_batch_num):
    model.eval()
    with torch.no_grad():
        model.module.reset_running_stats_for_calibration()
        for batch_idx, x in enumerate(train_loader):
            img = x['img'].data[0].cuda()
            img_metas = x['img_metas'].data[0]
            gt_semantic_seg = x['gt_semantic_seg'].data[0].cuda()
            if batch_idx >= post_bn_calibration_batch_num:
                break
            model(img=img, img_metas=img_metas, return_loss=True, gt_semantic_seg=gt_semantic_seg)  #forward only    

def validate_subnet(subnet, args, distributed, val_loader,
            eval_kwargs, eval_on_format_results):
    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    if not distributed:
        results = single_gpu_test(
            subnet, val_loader, args.show, args.show_dir, False, args.opacity,
            pre_eval=args.eval is not None and not eval_on_format_results,
            format_only=args.format_only or eval_on_format_results,
            format_args=eval_kwargs)
    else:
        results = multi_gpu_test(
            subnet, val_loader, args.tmpdir, args.gpu_collect, False, 
            pre_eval=args.eval is not None and not eval_on_format_results,
            format_only=args.format_only or eval_on_format_results, 
            format_args=eval_kwargs)
    return results

def is_pareto(k, benckmarks: list):
    for x in benckmarks:
        if k['flops'] > x['flops'] and k['mIoU'] < x['flops']:
            return False
    return True   

if __name__ == '__main__':
    main()