from aiearth.deeplearning.trainer.multispectral import MSClsTrainer
trainer=MSClsTrainer(work_dir="./work_dirs", config_name="swin_tiny_0.1_linear.yaml")
print(trainer.cfg)
trainer.train(validate=True)