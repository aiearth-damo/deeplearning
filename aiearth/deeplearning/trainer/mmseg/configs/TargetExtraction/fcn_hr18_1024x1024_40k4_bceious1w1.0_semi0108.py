_base_ = [
    "../_base_/models/fcn_hr18.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_20k.py",
]

# dataset settingsL
dataset_type = "RemoteSensingBinary"
data_root = ""
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
img_scale = (1024, 1024)
crop_size = (1024, 1024)
samples_per_gpu = 8
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(type="RandomROT90"),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
train_pipeline_semi = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(type="RandomROT90"),
    dict(
        type="CutMixSemi",
        pre_transforms=[dict(type="LoadImageFromFile"),
                        dict(type="LoadAnnotations")],
        p=1.0,
        alpha=1.0,
    ),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAugRS",
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

train = dict(
    type=dataset_type,
    data_root=data_root,
    img_dir="images",
    ann_dir="annotations",
    split="lst/train_old.txt",
    pipeline=train_pipeline,
)
train_semi = dict(
    type=dataset_type,
    data_root="",
    img_dir="images",
    ann_dir="results_semi0108",
    split="lst/train_semi.txt",
    pipeline=train_pipeline_semi,
)

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=4,
    train=[train, train_semi],
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images",
        ann_dir="annotations",
        split="lst/val_0121.txt",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images",
        ann_dir="annotations",
        split="lst/val_old.txt",
        pipeline=test_pipeline,
    ),
)

model = dict(
    type="EncoderDecoderBinary",
    pretrained="open-mmlab://msra/hrnetv2_w18_small",
    backbone=dict(
        extra=dict(
            stage1=dict(num_blocks=(2,)),
            stage2=dict(num_blocks=(2, 2)),
            stage3=dict(num_modules=3, num_blocks=(2, 2, 2)),
            stage4=dict(num_modules=2, num_blocks=(2, 2, 2, 2)),
        )
    ),
    decode_head=dict(
        num_classes=1,
        loss_decode=dict(
            type="EnsembleLoss",
            loss_dict_list=[
                dict(type="BCELoss", loss_weight=1.0, removeIgnore=True),
                dict(
                    type="BinaryDiceLoss",
                    batch=True,
                    loss_weight=1.0,
                    sigmoid=True,
                    iou=True,
                    smooth=1.0,
                ),
            ],
        ),
    ),
    test_cfg=dict(mode="whole_sigmoid", thresh=0.5),
)
