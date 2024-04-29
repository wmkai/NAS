# Copyright (c) OpenMMLab. All rights reserved.
import os
import random
import warnings

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         build_runner, get_dist_info)
from mmcv.utils import build_from_cfg

from mmseg import digit_version
from mmseg.core import BigNASDistEvalHook, BigNASEvalHook, build_optimizer
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import (build_ddp, build_dp, find_latest_checkpoint,
                         get_root_logger)


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

# 这个函数就是我们这个文件的核心函数了！！！
# 可以看到这个函数传进来了：
# 模型model，数据集dataset，配置cfg，是否分布式distributed，
# 是否验证validate，时间戳timestamp，基础环境信息等meta
def bignas_train_segmentor(model,
                        dataset,
                        cfg,
                        distributed=False,
                        validate=False,
                        timestamp=None,
                        meta=None):
    """Launch segmentor training."""
    # 获取日志对象，get_root_logger函数就是获取一个logger对象，之后打印输出信息都是用它
    # 我们看到，函数指定了log_level=cfg.log_level，也就是日志等级，通常为INFO，这里不了解的
    # 也可以不管，只需要知道这里获取了一个用于之后我们打印输出信息的日志对象logger就行。
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    # mmdet为了多个数据集一起训练，设置了这种dataset列表，
    # 往往我们只需要用到一个数据集，也就是一个dataset，所以就当在外面套了个列表就行
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        drop_last=True)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })

    # The specific dataloader settings
    # 构建dataloader，我们之前已经把dataset传进来了，这里就是实例化dataloader
    # 等价于 data_loaders = DataLoader(datasets, ...)
    # 现在先这么理解就对了，后面再详细介绍他是怎么build的，
    # 可以看出，指定了dataset，batch_size,work_num等参数，
    # 这些也都是我们平时使用pytorch构建dataloader时的基本套路
    train_loader_cfg = {**loader_cfg, **cfg.data.get('train_dataloader', {})}
    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset] # 是个list，可能会有多个数据集

    # put model on devices
    # 将模型放到GPU上，这里分为分布式和非分布式，这里强调一下，mmdet的非分布式训练只支持单卡
    # 想要多卡训练，就使用分布式训练
    # 可以看到，分布式时，就将模型送入cuda，然后用MMDistributedDataParallel包起来，
    # 我们先不管MMDistributedDataParallel到底是什么，用过pytorch的同学都知道DistributedDataParallel
    # 这个类，他就是用于分布式训练的，我们将MMDistributedDataParallel等价于DistributedDataParallel理解就行了。
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # DDP wrapper
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        # 非分布式时，将模型送入cuda，然后用MMDataParallel包起来，
        # 我们将MMDataParallel等价于pytorch的DataParallel理解就行。
        if not torch.cuda.is_available():
            assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                'Please use MMCV >= 1.4.4 for CPU training!'
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # build runner
    # 构建优化器，与之前一样，我们就把他视为optimizer = SGD(....)或者optimizer = Adam(...)就行，
    # 至于使用的什么优化器，哪些参数等，在配置文件中都有指定。
    optimizer = build_optimizer(model, cfg.optimizer)

    # 这里判断一些关于runner的配置信息，不重要，也很好理解，不过多解释。
    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    # 构建runner，关于runner，先简单解释一些，他就是一个运行器，包含一堆和训练有关的属性和函数，
    # 与上面相同，就是根据cfg中的配置信息实例化一个runner对象，
    # 后续我会专门出一篇文章，详细介绍runner是什么以及是干什么的，怎么运行的等等。
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    # 如果是分布式训练，并且用的是EpochBasedRunner，就要注册一个DistSamplerSeedHook，
    # 这个主要是用来设置分布式训练中data sampler中的随机种子的；
    if distributed:
        # when distributed training by epoch, using`DistSamplerSeedHook` to set
        # the different seed to distributed sampler for each epoch, it will
        # shuffle dataset at each epoch and avoid overfitting.
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())
    
    # an ugly walkaround to make the .log and .log.json filenames the same
    # 把时间戳赋值给runner的timestamp属性
    runner.timestamp = timestamp

    # register eval hooks
    # 注册验证hook，我们训练过程中，往往是每隔几个epoch就跑一次验证，看看评估结果，
    # 这里就是处理验证相关的数据集等，往往 validate 为 True，
    # 可以看到if里面的各种操作和之前构建训练数据集时差不多。
    if validate:
        # 构建验证数据集和dataloader，与之前构建训练验证集和dataloader的逻辑相同，
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        # The specific dataloader settings
        val_loader_cfg = {
            **loader_cfg,
            'samples_per_gpu': 1,
            'shuffle': False,  # Not shuffle by default
            **cfg.data.get('val_dataloader', {}),
        }
        val_dataloader = build_dataloader(val_dataset, **val_loader_cfg)
        # 获取验证相关的一些配置信息，如果是分布式就注册DistEvalHook，如果不是就注册EvalHook，
        # 不管注册的是哪个hook，他们都是在训练过程中执行验证操作的，
        # 比如每隔5个epoch就跑一次val数据集的推理，然后评估结果，Eval hook就是完成这些操作的；
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = BigNASDistEvalHook if distributed else BigNASEvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook( # data_loaders是个list，可能会有多个数据集，具体可以看上面定义
            eval_hook(val_dataloader, train_loader=data_loaders[0],**eval_cfg), priority='LOW')

    # user-defined hooks
    # 注册一些自己定义的hook，我也没有自己定义什么hook，mmdet提供的hook已经满足我的需要了，
    # 所以下面这一堆不会执行，如果你也没有指定自己的custom_hooks，那就不会执行下面的if，
    # 也就不会注册自己定义的hook；
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
        if resume_from is not None:
            cfg.resume_from = resume_from
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
