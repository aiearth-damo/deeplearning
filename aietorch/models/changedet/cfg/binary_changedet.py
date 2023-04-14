# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='ChangedetEncoderDecoder',
    pretrained='open-mmlab://msra/hrnetv2_w18',
    backbone=dict(
        type='HRNet',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(18, 36)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144)))),
    neck=dict(
        type='ChangeDetCatBifpn',
        in_channels=[18, 36, 72, 144],
        num_channels=96,
    ),
    decode_head=dict(
        type='ChangeDetHead',
        in_channels=96,
        in_index=0,
        channels=16,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='DiceBceLoss'),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole_sigmoid', thresh=0.5, semi_probs=[]))
