_base_ = "./fcn_hr18_1024x1024_16k_landcover.py"
runner = dict(type="IterBasedRunner", max_iters=90000)
checkpoint_config = dict(by_epoch=False, interval=30000)
evaluation = dict(interval=5000, metric="mIoU")
data = dict(samples_per_gpu=2, workers_per_gpu=2)
model = dict(
    pretrained="open-mmlab://msra/hrnetv2_w48",
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)),
        )
    ),
    decode_head=dict(
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384]), num_classes=11
    ),
)
