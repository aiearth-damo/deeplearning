# dataset settings
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="LandcoverLoader",
        data_root="",
        img_dir="",
        ann_dir="",
        split="",
        reduce_zero_label=False,
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations"),
            dict(type="Resize", img_scale=(1024, 1024), ratio_range=(0.5, 2.0)),
            dict(type="RandomCrop", crop_size=(1024, 1024), cat_max_ratio=0.75),
            dict(type="RandomFlip", prob=0.5),
            dict(type="PhotoMetricDistortion"),
            dict(
                type="Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[79.6875, 79.6875, 79.6875],
                to_rgb=True,
            ),
            dict(type="Pad", size=(1024, 1024), pad_val=0, seg_pad_val=255),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_semantic_seg"]),
        ],
    ),
    val=dict(
        type="LandcoverLoader",
        data_root="",
        img_dir="",
        ann_dir="",
        split="",
        reduce_zero_label=False,
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=(1024, 1024),
                # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    # dict(type='RandomFlip'),
                    dict(
                        type="Normalize",
                        mean=[127.5, 127.5, 127.5],
                        std=[79.6875, 79.6875, 79.6875],
                        to_rgb=True,
                    ),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ],
    ),
    test=dict(
        type="LandcoverLoader",
        data_root="",
        img_dir="",
        ann_dir="",
        split="",
        reduce_zero_label=False,
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=(1024, 1024),
                # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    # dict(type='RandomFlip'),
                    dict(
                        type="Normalize",
                        mean=[127.5, 127.5, 127.5],
                        std=[79.6875, 79.6875, 79.6875],
                        to_rgb=True,
                    ),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ],
    ),
)