# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, bignas_train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import (collect_env, get_device, get_root_logger,
                         setup_multi_processes)

# 此函数是解析命令行输入的参数的，先不看他，我们先找到最下面的if __name__ == '__main__'处，
# 此函数中的各种参数我们先不解释，当用到他们的时候，在哪里用到就在哪里解释
def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff_seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')
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
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args


def main():
    # 来到main这里，我们可以看到，
    # 首先运行的是上面的parse_args函数，也就是先把命令行的参数解析出来
    args = parse_args()

    # 下面一句是从args.config参数指定的配置文件名称中读取配置。
    # 例如：args.config = './configs/centernet/centernet_resnet18_dcnv2_140e_coco.py
    # centernet_resnet18_dcnv2_140e_coco.py文件中有一个配置为dataset_type = 'CocoDataset'
    # 那么Config.fromfile()函数将从centernet_resnet18_dcnv2_140e_coco.py文件中加载配置，
    # cfg就会有一个属性dataset_type，即：cfg.dataset_type = 'CocoDataset'
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options) # 这一句是将args.cfg_options中的配置加载到cfg中

    # set cudnn_benchmark
    # 此处为设置是否使用cudnn加速
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    # 设置工作目录，也就是你训练过程中产生的各种checkpoint等文件存放的地方
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    # 设置从哪个checkpoint重新开始训练
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    # 如果没有指定gpu_ids，指定了gpus这个参数，这个代表使用的gpu数量，
    # 如果gpus为2，那么cfg.gpu_ids为[0，1]
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    # 设置使用的gpu的id，比如使用两块GPU，编号分别为 1，2，则args.gpu_ids应设置为1，2
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    cfg.auto_resume = args.auto_resume

    # init distributed env first, since logger depends on the dist info.
    # 设置是否分布式训练，args.launcher不指定就是不使用分布式，否则就是分布式训练
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params) # 初始化pytorch的分布式
        # gpu_ids is used to calculate iter when resuming checkpoint
        # 获取分布式信息，返回的是当前进程的id：rank，和总进程数量world_size
        _, world_size = get_dist_info()
        # 总进程数量也就是要使用的gpu数量，如果world_size为4，则使用的gpu_ids为[0,1,2,3]
        cfg.gpu_ids = range(world_size)

    # create work_dir
    # 前面只是指定了工作目录，这里要创建工作目录，osp.abspath(cfg.work_dir)是获取绝对路径
    # mmcv.mkdir_or_exist是检查目录是否存在，不存在就创建他
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    # 把配置信息写入到文件中
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    # 初始化日志对象logger，之后程序打印的任何信息都是logger去打印的
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # set multi-process settings
    setup_multi_processes(cfg)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    # 初始化meta字典来记录一些重要的信息，比如环境变量之类的，把他们打印到日志中
    meta = dict()
    # log env info
    env_info_dict = collect_env() # 收集系统环境信息
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line) # 打印系统环境信息
    meta['env_info'] = env_info 

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}') # 记录配置文件的名称

    # set random seeds
    # 设置随机种子
    cfg.device = get_device()
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    # 构建segmentor
    # 假如我们已经定义了一个ModelNet(nn.Module)模型
    # 那下面这句话就相当于 model = ModelNet()，只不过他是根据配置文件中的信息去构建网络
    # 追根溯源就等价于model = ModelNet()
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights() # 初始化模型参数，init_weights函数是定义在model中的，以后再看model的源码

    # SyncBN is not support for DP
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        model = revert_sync_batchnorm(model)

    logger.info(model)

    # 构建Dataset
    # 和上面model一样，下边这一句也就相当于 datasets = torch.utils.data.Dataset(...)
    # 只不过他用列表包起来了，不必纠结。
    datasets = [build_dataset(cfg.data.train)]
    # cfg.workflow是指执行流，他可以指定边训练边验证
    # 比如训练两个epoch，就验证一次，那么cfg.workflow = [('train', 2), ('val', 1)]
    # 如果len(cfg.workflow) == 2就说明有验证部分，
    # 但通常情况下，我们都设置cfg.workflow = [('train', 1)]，也就是下面的if进不去
    if len(cfg.workflow) == 2:
        # 构建验证数据集
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    # checkpoint配置，在checkpoint配置中加一些基础版本信息和检测类别信息
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    # 将数据集的类别赋值给模型
    model.CLASSES = datasets[0].CLASSES
    # passing checkpoint meta for saving best checkpoint
    meta.update(cfg.checkpoint_config.meta)
    # 到apis/train.py中训练segmentor，下面这个train_segmentor函数将会在做一些训练前的准备之后开始训练
    bignas_train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
