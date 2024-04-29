# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
import os
import torch
import torch.distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm

import mmcv
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
# from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmseg import digit_version
from mmseg.utils import get_root_logger, build_ddp, build_dp, get_device

def bn_calibration(model, train_loader, post_bn_calibration_batch_num):
    model.eval()
    with torch.no_grad():
        model.module.reset_running_stats_for_calibration() # 把bn层的.training设置为True,.momentum设置为None,并调用reset_running_stats()
        for batch_idx, x in enumerate(train_loader):
            img = x['img'].data[0].cuda()
            img_metas = x['img_metas'].data[0]
            gt_semantic_seg = x['gt_semantic_seg'].data[0].cuda()
            if batch_idx >= post_bn_calibration_batch_num:
                break
            model(img=img, img_metas=img_metas, return_loss=True, gt_semantic_seg=gt_semantic_seg)  #forward only            

class BigNASEvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self,
                 *args, # val_loader传入args调用父类构造函数
                 train_loader=None,
                 post_bn_calibration_batch_num=64,
                 by_epoch=False,
                 efficient_test=False,
                 pre_eval=False,
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.train_loader = train_loader
        self.post_bn_calibration_batch_num = post_bn_calibration_batch_num
        self.pre_eval = pre_eval
        if efficient_test:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` for evaluation hook '
                'is deprecated, the evaluation hook is CPU memory friendly '
                'with ``pre_eval=True`` as argument for ``single_gpu_test()`` '
                'function')

    def _do_evaluate(self, runner):
        logger = get_root_logger()
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return
        
        subnets_to_be_evaluated = { 
            'max_net': {},
            'min_net': {},
        }
        for net_id in subnets_to_be_evaluated:
            if net_id == 'min_net': 
                runner.model.module.sample_min_subnet()
            elif net_id == 'max_net': 
                runner.model.module.sample_max_subnet()
            elif net_id.startswith('random_net'):
                runner.model.module.sample_active_subnet()
            else:
                runner.model.module.set_active_subnet(
                    subnets_to_be_evaluated[net_id]['width'],
                    subnets_to_be_evaluated[net_id]['depth'],
                    subnets_to_be_evaluated[net_id]['kernel_size'],
                    subnets_to_be_evaluated[net_id]['expand_ratio'],
                    subnets_to_be_evaluated[net_id]['num_heads'],
                    subnets_to_be_evaluated[net_id]['key_dim'],
                    subnets_to_be_evaluated[net_id]['attn_ratio'],
                    subnets_to_be_evaluated[net_id]['mlp_ratio'],
                    subnets_to_be_evaluated[net_id]['transformer_depth']
                )
            if runner.rank == 0:
                logger.info(net_id + ' val result:')
            subnet = runner.model.module.get_active_subnet()            
            subnet_cfg = runner.model.module.get_active_subnet_settings()
            if runner.rank == 0:
                logger.info('Subnet cfg: ')
                for k, v in subnet_cfg.items():
                    logger.info(k + ': ' + str(v))
                # estimate running mean and running statistics
                logger.info('start bn_calibration...')

            subnet = revert_sync_batchnorm(subnet) # 把syncbn的模型转为bn
            subnet = build_dp(subnet, get_device(), device_ids=[0])

            bn_calibration(subnet, self.train_loader, self.post_bn_calibration_batch_num)
            if runner.rank == 0:
                logger.info('finish bn_calibration...')

            from mmseg.apis import single_gpu_test
            results = single_gpu_test(
                subnet, self.dataloader, show=False, pre_eval=self.pre_eval)
            runner.log_buffer.clear()
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)
            if self.save_best and net_id == 'max_net':
                self._save_ckpt(runner, key_score)


class BigNASDistEvalHook(_DistEvalHook): # 默认是分布式测试
    """Distributed EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self,
                 *args, # val_loader传入args调用父类构造函数
                 train_loader=None,
                 post_bn_calibration_batch_num=64,
                 by_epoch=False,
                 efficient_test=False,
                 pre_eval=False,
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.train_loader = train_loader
        self.post_bn_calibration_batch_num = post_bn_calibration_batch_num
        self.pre_eval = pre_eval
        if efficient_test:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` for evaluation hook '
                'is deprecated, the evaluation hook is CPU memory friendly '
                'with ``pre_eval=True`` as argument for ``multi_gpu_test()`` '
                'function')

    def _do_evaluate(self, runner):
        logger = get_root_logger()
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        subnets_to_be_evaluated = { 
            'max_net': {},
            'min_net': {},
        }
        # 这里runner.model是MMDistributedDataParallel
        with torch.no_grad():
            for net_id in subnets_to_be_evaluated:
                if net_id == 'min_net': 
                    runner.model.module.sample_min_subnet()
                elif net_id == 'max_net': 
                    runner.model.module.sample_max_subnet()
                elif net_id.startswith('random_net'):
                    runner.model.module.sample_active_subnet()
                else:
                    runner.model.module.set_active_subnet(

                        subnets_to_be_evaluated[net_id]['width'],
                        subnets_to_be_evaluated[net_id]['depth'],
                        subnets_to_be_evaluated[net_id]['kernel_size'],
                        subnets_to_be_evaluated[net_id]['expand_ratio'],
                        subnets_to_be_evaluated[net_id]['num_heads'],
                        subnets_to_be_evaluated[net_id]['key_dim'],
                        subnets_to_be_evaluated[net_id]['attn_ratio'],
                        subnets_to_be_evaluated[net_id]['mlp_ratio'],
                        subnets_to_be_evaluated[net_id]['transformer_depth']
                    )
                if runner.rank == 0:
                    logger.info(net_id + ' val result:')
                subnet = runner.model.module.get_active_subnet(preserve_weight=True)
                # subnet是StaticEncoderDecoder类，没有被MMDistributedDataParallel包装，已经在gpu上
                subnet_cfg = runner.model.module.get_active_subnet_settings()
                if runner.rank == 0:
                    logger.info('Subnet cfg: ')
                    for k, v in subnet_cfg.items():
                        logger.info(k + ': ' + str(v))
                    # estimate running mean and running statistics
                    logger.info('start bn_calibration...')

                # DDP wrapper
                if 'LOCAL_RANK' not in os.environ:
                    os.environ['LOCAL_RANK'] = str(runner.rank)
                subnet = build_ddp(
                    subnet,
                    get_device(),
                    device_ids=[int(os.environ['LOCAL_RANK'])],
                    broadcast_buffers=False,
                    find_unused_parameters=runner.model.find_unused_parameters)
            
                if self.broadcast_bn_buffer:
                    for name, module in subnet.named_modules():
                        if isinstance(module,
                                    _BatchNorm) and module.track_running_stats:
                            dist.broadcast(module.running_var, 0) # 把rank0的running参数发送到其他卡上
                            dist.broadcast(module.running_mean, 0)

                if not self._should_evaluate(runner):
                    return

                tmpdir = self.tmpdir
                if tmpdir is None:
                    tmpdir = osp.join(runner.work_dir, '.bignas_eval_hook')

                bn_calibration(subnet, self.train_loader, self.post_bn_calibration_batch_num)
                if runner.rank == 0:
                    logger.info('finish bn_calibration...')

                from mmseg.apis import multi_gpu_test
                results = multi_gpu_test(
                    subnet,
                    self.dataloader,
                    tmpdir=tmpdir,
                    gpu_collect=self.gpu_collect,
                    pre_eval=self.pre_eval)

                runner.log_buffer.clear()

                if runner.rank == 0:
                    print('\n')
                    runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
                    key_score = self.evaluate(runner, results) # 只用于打印log

                    if self.save_best and net_id == 'max_net':
                        self._save_ckpt(runner, key_score)
