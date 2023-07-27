_base_ = [
    "../_base_/models/fcn_hr18.py",
    "../_base_/datasets/landcover.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_160k.py",
]
model = dict(type="EncoderDecoderLandcover", decode_head=dict(num_classes=11))
runner = dict(type="IterBasedRunner", max_iters=22500)
checkpoint_config = dict(by_epoch=False, interval=7500)
evaluation = dict(interval=2250, metric="mIoU")
optimizer_config = dict(type="Fp16OptimizerHook", loss_scale=512.0)
fp16 = dict()
