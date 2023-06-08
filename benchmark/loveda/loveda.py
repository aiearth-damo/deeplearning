from aiearth.deeplearning.trainer.mmseg import LandcoverTrainer


if __name__ == '__main__':
    work_dir = "./work_dirs"
    
    # model
    trainer = LandcoverTrainer(work_dir=work_dir)
    trainer.set_base_cfg_from_file("config/fcn_hr18_512x512_80k_loveda.py")
    
    trainer.train(validate=True)
    
