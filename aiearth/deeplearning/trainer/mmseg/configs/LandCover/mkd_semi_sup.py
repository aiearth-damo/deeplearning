# dataset settingsL
dataset_type = "LandcoverLoader"
data_root = "/mnt_js/jiasheng.tjs/data/landcover_data/"
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[79.6875, 79.6875, 79.6875], to_rgb=True
)
crop_size = (1024, 1024)
img_scale = (1024, 1024)
sup_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=img_scale, ratio_range=(0.75, 1.5)),
    dict(type="RandomCrop", crop_size=crop_size),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="GenerateCutBox", prop_range=[0.25, 0.5], n_boxes=3, crop_size=crop_size),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg", "cutmask"]),
]


unsup_train_pipeline = [
    dict(
        type="StrongWeakAug",
        pre_transforms=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations"),
            dict(type="Resize", img_scale=img_scale, ratio_range=(0.5, 2.0)),
            dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=1.0),
            dict(type="RandomFlip", prob=0.5, direction="horizontal"),
            dict(type="RandomFlip", prob=0.5, direction="vertical"),
        ],
        weak_transforms=[
            dict(type="PhotoMetricDistortion"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
            dict(
                type="GenerateCutBox",
                prop_range=[0.25, 0.5],
                n_boxes=3,
                crop_size=crop_size,
            ),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_semantic_seg", "cutmask"]),
        ],
        strong_transforms=[
            dict(
                type="Albu",
                transforms=[
                    dict(
                        type="SomeOf",
                        n=1,
                        p=1.0,
                        transforms=[
                            dict(
                                type="ColorJitter",
                                brightness=0.5,
                                contrast=0.5,
                                saturation=0.5,
                                hue=0.25,
                                p=0.3,
                            ),
                            dict(
                                type="GaussianBlur",
                                blur_limit=(3, 7),
                                sigma_limit=(0.1, 2.0),
                                p=0.5,
                            ),
                            dict(type="Equalize", p=0.1),
                            dict(type="Solarize", p=0.1),
                            dict(type="ToGray", p=0.5),
                        ],
                    ),
                ],
            ),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
            dict(
                type="GenerateCutBox",
                prop_range=[0.25, 0.5],
                n_boxes=3,
                crop_size=crop_size,
            ),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_semantic_seg", "cutmask"]),
        ],
    )
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
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

classes = [
    "背景",
    "industrial_land",
    "garden_land",
    "urban_residential",
    "arbor_forest",
    "rural_residential",
    "shrub_land",
    "traffic_land",
    "natural_meadow",
    "paddy_field",
    "artificial_meadow",
    "irrigated_land",
    "river",
    "dry_cropland",
    "lake",
    "pond",
]
train_sup = dict(
    type=dataset_type,
    data_root=data_root,
    img_dir="",
    ann_dir="",
    split="train_v1.6_10_sup.txt",
    reduce_zero_label=False,
    ignore_index=255,
    pipeline=sup_train_pipeline,
)

train_unsup = dict(
    type=dataset_type,
    data_root=data_root,
    img_dir="",
    ann_dir="",
    split="train_v1.6_10_unsup.txt",
    reduce_zero_label=False,
    ignore_index=255,
    pipeline=unsup_train_pipeline,
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type="SemiDataset",
        sup_dataset=train_sup,
        unsup_dataset=train_unsup,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="",
        ann_dir="",
        split="test_v1.6.txt",
        reduce_zero_label=False,
        ignore_index=255,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="",
        ann_dir="",
        split="test_v1.6.txt",
        reduce_zero_label=False,
        ignore_index=255,
        pipeline=test_pipeline,
    ),
)


algorithm = dict(
    type="ExpGeneralSemiCPSFDOnlineEMAOnce3CutmixSWFD",
    architecture=dict(type="MMSegArchitecture", model="same"),
    end_momentum=0.4,
    components=[
        dict(
            module="decode_head.conv_seg",
            losses=[
                dict(
                    type="ExpSemiLossCPSFAWS7",
                    name="loss_semi",
                    loss_weight=1.5,
                    loss_weight2=1.0,
                    # avg_non_ignore=True,
                    ignore_index=255,
                    total_iteration=None,
                    align_corners=False,
                    branch1=False,
                    branch2=False,
                    teacher1=False,
                    teacher2=False,
                    end_ratio=0.0,
                    scale_factor=None,
                    thresh=0.0,
                    ratio_type=None,
                )
            ],
        )
    ],
)
