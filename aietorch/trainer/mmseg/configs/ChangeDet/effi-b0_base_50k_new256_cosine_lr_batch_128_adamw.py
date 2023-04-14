# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
model = dict(
    type="ChangedetEncoderDecoder",
    backbone=dict(
        type="EfficientNet",
        config="effb0",
    ),
    neck=dict(
        type="ChangeDetCatBifpn",
        in_channels=[24, 40, 80, 192],
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
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole_sigmoid", thresh=0.5)
)

# dataset settings
dataset_type = "ChangeDetDataset"
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
    dict(type="DoubleImageLoadAsBinaryLabel"),
    dict(type="DoubleImagePad", size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type="DefaultFormatBundleDoubleImage"),
    dict(type="Collect", keys=["img1", "img2", "gt_semantic_seg"]),
]
test_pipeline = [
    dict(type="LoadDoubleImageFromFile"),
    dict(
        type="MultiScaleFlipAugCD",
        # img_scale=(1024, 1024),
        img_scale=None,
        img_ratios=[1.0],
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
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        img_dir="",
        ann_dir="",
        binary_label=True,
        split="",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        img_dir="",
        ann_dir="",
        binary_label=True,
        split="",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        img_dir="",
        ann_dir="",
        binary_label=True,
        split="",
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
# resume_from = (
#     "work_dirs/changedet/effi-b0_base_50k_new256_cosine_lr_batch_128_adamw/latest.pth"
# )
workflow = [("train", 1)]
cudnn_benchmark = True

# optimizer
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.0005)
optimizer_config = dict(type="Fp16OptimizerHook", loss_scale="dynamic")
# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    min_lr=1e-7,
    by_epoch=False,
    warmup="linear",
    warmup_iters=1000,
    warmup_ratio=1.0 / 100,
)
# runtime settings
runner = dict(type="IterBasedRunner", max_iters=50000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(
    interval=2000,
    metric=["mIoU", "mFscore"],
    save_best="IoU.changedarea",
    rule="greater",
)
find_unused_parameters = True
