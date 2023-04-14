# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
model = dict(
    type="ChangedetEncoderDecoder",
    # pretrained='open-mmlab://resnet50_v1c',
    pretrained="open-mmlab://msra/hrnetv2_w18",
    backbone=dict(
        type="HRNet",
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block="BOTTLENECK",
                num_blocks=(4,),
                num_channels=(64,),
            ),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block="BASIC",
                num_blocks=(4, 4),
                num_channels=(18, 36),
            ),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block="BASIC",
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72),
            ),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block="BASIC",
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144),
            ),
        ),
    ),
    neck=dict(
        type="ChangeDetCatBifpn",
        in_channels=[18, 36, 72, 144],
        num_channels=96,
    ),
    decode_head=dict(
        sampler=dict(type="OHEMPixelSampler", thresh=0.7, min_kept=100000),
        type="BuildingChangeHead",
        in_channels=96,
        in_index=0,
        channels=16,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=4,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss"),
        # loss_decode=dict(type='DiceLoss')
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

# dataset settings
dataset_type = "BuildingChangeDataset"
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[79.6875, 79.6875, 79.6875], to_rgb=False
)
crop_size = (896, 896)
train_pipeline = [
    dict(type="LoadDoubleImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="DoubleImageResize", img_scale=(
        1024, 1024), ratio_range=(0.5, 2.0)),
    dict(type="DoubleImageRandomCrop", crop_size=crop_size, cat_max_ratio=1.0),
    dict(type="DoubleImageRandomFlip", prob=0.5),
    dict(type="DoubleImagePhotoMetricDistortion"),
    dict(type="DoubleImageRandomRgbBgr"),
    dict(type="DoubleImageNormalize", **img_norm_cfg),
    dict(type="DoubleImagePad", size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type="DefaultFormatBundleDoubleImage"),
    dict(type="Collect", keys=["img1", "img2", "gt_semantic_seg"]),
]
test_pipeline = [
    dict(type="LoadDoubleImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(896, 896),
        flip=False,
        transforms=[
            dict(type="DoubleImageResize", keep_ratio=True),
            # dict(type='DoubleImageRandomFlip'),
            dict(type="DoubleImageNormalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img1", "img2"]),
            dict(type="Collect", keys=["img1", "img2"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        img_dir="",
        ann_dir="",
        binary_label=False,
        split="",
        classes=4,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        img_dir="",
        ann_dir="",
        binary_label=False,
        split="",
        classes=4,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        img_dir="",
        ann_dir="",
        binary_label=False,
        split="",
        classes=4,
        pipeline=test_pipeline,
    ),
)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook"),
    ],
)
# yapf:enable
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = ""
resume_from = None
workflow = [("train", 1)]
cudnn_benchmark = True

# optimizer
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy="CosineAnnealing", min_lr=1e-5, by_epoch=False)
# runtime settings
runner = dict(type="IterBasedRunner", max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=10000)
# evaluation = dict(interval=2000, metric='mIoU')
evaluation = dict(interval=10000, metric="mIoU")
find_unused_parameters = True
