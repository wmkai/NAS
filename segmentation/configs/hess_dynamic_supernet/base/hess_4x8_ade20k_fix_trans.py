_base_ = [
    '../../_base_/datasets/ade20k.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_160k.py',
    './hess_supernet.py'
]
model = dict(backbone= dict(fix_backbone=True, 
                            fix_trans=False, 
                            supernet=dict(pretrained='ckpt/hess_supernet/base/imagenet/fix_trans/last.pth.tar')))

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
    interval=50, 
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False)
    ]
) 
data=dict(samples_per_gpu=4)
find_unused_parameters = True

# evaluation = dict(interval=50, metric='mIoU', pre_eval=True) # for debug
evaluation = dict(interval=16000, metric='mIoU', pre_eval=True) # 32000