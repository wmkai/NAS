_base_ = [
    '../../_base_/datasets/ade20k.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_160k.py',
    './hess_supernet.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(backbone=dict(fix_backbone=False, fix_trans=False, supernet=dict(SIM=dict(norm_cfg=norm_cfg))),
             decode_head=dict(norm_cfg=norm_cfg),
             sync_bn=False)
constraint_flops=1.8
unfixed_ckpt='ckpt/hess_supernet/base/ade20k/unfixed/latest.pth'
fix_backbone_ckpt='ckpt/hess_supernet/base/ade20k/fix_backbone/latest.pth'
fix_trans_ckpt='ckpt/hess_supernet/base/ade20k/fix_trans/latest.pth'
post_bn_calibration_batch_num=64
data_root = 'data/ade/ADEChallengeData2016'
data=dict(train=dict(data_root=data_root),
          val=dict(data_root=data_root),
          test=dict(data_root=data_root),
          samples_per_gpu=4)
runner = dict(type='IterBasedRunner', max_iters=1500) # 采样子网后finetune
need_finetune=False

optimizer = dict(_delete_=True, type='AdamW', lr=0.00012, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
log_config = dict(
    _delete_=True,
    interval=500, 
    hooks=[dict(type='TextLoggerHook', by_epoch=False),]
) 
find_unused_parameters = True

# evaluation = dict(interval=50, metric='mIoU', pre_eval=True) # for debug
evaluation = dict(interval=16000, metric='mIoU', pre_eval=True) # 32000
work_dir = 'output/hess_supernet/base/unfixed'

