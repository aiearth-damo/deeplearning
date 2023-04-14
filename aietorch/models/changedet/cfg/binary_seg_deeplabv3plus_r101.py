# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoderBinary',
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="EnsembleLoss",
                         loss_dict_list=[
                             dict(type='BCELoss', loss_weight=1.0, removeIgnore=True),
                             dict(type='BinaryDiceLoss', batch=True, loss_weight=1.0, sigmoid=True, iou=True,
                                  smooth=1.0),
                         ])),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="EnsembleLoss",
                         loss_dict_list=[
                             dict(type='BCELoss', loss_weight=0.4, removeIgnore=True),
                             dict(type='BinaryDiceLoss', batch=True, loss_weight=0.4, sigmoid=True, iou=True,
                                  smooth=1.0),
                         ])),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole_sigmoid', thresh=0.5))
