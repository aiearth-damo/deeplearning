# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
model = dict(
    type="ChangedetEncoderDecoder",
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
        type="ChangeDetHead",
        in_channels=96,
        in_index=0,
        channels=16,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="DiceBceLoss"),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole_sigmoid", thresh=0.5, semi_probs=[]),
)

# dataset settings
dataset_type = "ChangeDetDataset"
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[79.6875, 79.6875, 79.6875], to_rgb=True
)
crop_size = (896, 896)
train_pipeline = [
    dict(type="LoadDoubleImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="DoubleImageResize", img_scale=(
        1024, 1024), ratio_range=(0.5, 2.0)),
    dict(type="DoubleImageRandomCrop", crop_size=crop_size, cat_max_ratio=1.0),
    dict(type="DoubleImageRandomFog", prob=0.05, size=400, random_color=True),
    dict(type="DoubleImageRandomFlip", prob=0.5),
    dict(type="DoubleImagePhotoMetricDistortion"),
    dict(type="DoubleImageRandomRgbBgr", prob=0.3),
    dict(type="DoubleImageNormalize", **img_norm_cfg),
    dict(type="DoubleImageLoadAsBinaryLabel"),
    dict(type="DoubleImagePad", size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type="DefaultFormatBundleDoubleImage"),
    dict(type="Collect", keys=["img1", "img2", "gt_semantic_seg"]),
]
test_pipeline = [
    dict(type="LoadDoubleImageFromFile"),
    dict(
        type="MultiScaleFlipAugCD",
        img_scale=None,
        img_ratios=[1.0],  # 多尺度测试
        # flip=False,  # flip 测试
        transforms=[
            dict(type="DoubleImageResize", keep_ratio=True),
            # dict(type='DoubleImageRandomFlip'),  # flip 测试
            dict(type="DoubleImageNormalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img1", "img2"]),
            dict(type="Collect", keys=["img1", "img2"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=[
        dict(
            type=dataset_type,
            img_dir="",
            ann_dir="",
            data_root="",
            semi_root=None,
            binary_label=True,
            split="changeDet/lst/trainx20.txt",
            pipeline=train_pipeline,
        ),
        dict(
            type=dataset_type,
            img_dir="",
            ann_dir="",
            data_root="",
            semi_root=None,
            binary_label=True,
            split="changeDet/lst/trainx20.txt",
            pipeline=train_pipeline,
        ),
    ],
    val=dict(
        type=dataset_type,
        img_dir="",
        ann_dir="",
        data_root="",
        semi_root=None,
        binary_label=True,
        split="changeDet/lst/train.txt",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        img_dir="",
        ann_dir="",
        data_root="",
        semi_root=None,
        binary_label=True,
        split="changeDet/lst/train.txt",
        pipeline=test_pipeline,
    ),
)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook"),
    ],
)
# yapf:enable
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
cudnn_benchmark = True

# optimizer
optimizer = dict(type="SGD", lr=0.005, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(type="Fp16OptimizerHook", loss_scale=512.0)
fp16 = dict()
# learning policy
lr_config = dict(policy="CosineAnnealing", min_lr=1e-5, by_epoch=False)
# runtime settings
runner = dict(type="IterBasedRunner", max_iters=150000)
checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(
    interval=5000,
    metric=["mIoU", "mFscore"],
    save_best="IoU.changedarea",
    rule="greater",
    pre_eval=True,
)
find_unused_parameters = True
