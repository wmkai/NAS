# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

model_cfgs = dict(
    stem_channel=24,
    cfg=[
        # k,  t,  c, s
        [3,   1,  16, 1], # 1/2        
        [5,   3,  32, 2], # 1/4 1                 
        [5,   4,  56, 2], # 1/8 3              
        [5,   4,  112, 2], # 1/16 5     
        [5,   4,  112, 1], #           
        [5,   5,  168, 2], # 1/32 7   
        [5,   5,  168, 1], #  
        [5,   5,  168, 1], #
        [5,   5,  168, 1], #                      
    ],
    channels=[32, 56, 112, 168],
    out_channels=[None, 256, 256, 256],
    embed_out_indice=[1, 2, 4, 8],
    decode_out_indices=[1, 2, 3],
    key_dim=[12, 18, 18, 20],
    num_heads=[12, 10, 12, 4],
    attn_ratios=[1.8, 2.4, 2.2, 2.2],
    mlp_ratios=[2.0, 2.2, 2.2, 2.2],
    c2t_stride=2,
)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='HESS',
        cfgs=model_cfgs['cfg'], 
        stem_channel=model_cfgs['stem_channel'], 
        channels=model_cfgs['channels'],
        out_channels=model_cfgs['out_channels'], 
        embed_out_indice=model_cfgs['embed_out_indice'],
        decode_out_indices=model_cfgs['decode_out_indices'],
        depths=[2, 1, 2, 2],
        key_dim=model_cfgs['key_dim'],
        num_heads=model_cfgs['num_heads'],
        attn_ratios=model_cfgs['attn_ratios'],
        mlp_ratios=model_cfgs['mlp_ratios'],
        c2t_stride=model_cfgs['c2t_stride'],
        drop_path_rate=0.1,
        norm_cfg=norm_cfg,
        init_cfg=dict(
            type='Pretrained', checkpoint='ckpt/search_5k/model_best.pth.tar')
    ),
    decode_head=dict(
        type='SimpleHead',
        in_channels=[256, 256, 256],
        in_index=[0, 1, 2],
        channels=256,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))