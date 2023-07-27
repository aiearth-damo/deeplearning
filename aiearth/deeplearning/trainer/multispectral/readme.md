# 该project提供基于s12的多光谱预训练模型及在下游数据集上的测试过程

## 关于多光谱预训练模型的说明
所有的模型地址都在configs中的配置文件中进行路径修改“pretrained”关键词。

预训练模型：
|  模型 | Uri |
| --- | --- |
| B13_rn50_moco_0099_ckpt.pth | aie://Multispectral/B13_rn50_moco_0099_ckpt.pth |
| B13_swin_tiny_0099_ckpt.pth | aie://Multispectral/B13_swin_tiny_0099_ckpt.pth |
| B13_vit_small_0099_ckpt.pth | aie://Multispectral/B13_vit_small_0099_ckpt.pth |

## 目前支持：
### Res50+Bigearthnet
   该模式下支持完全finetune，线性finetune和测试三个过程：
   
   完全fientune: 性能 91.2%
   
    from aiearth.deeplearning.trainer.multispectral import MSClsTrainer
    trainer=MSClsTrainer(work_dir="./work_dirs", config_name="r50_1.0_finetune.yaml")
    print(trainer.cfg)
    trainer.train(validate=True)
   
   线性finetune: 性能 82.2%
   
    from aiearth.deeplearning.trainer.multispectral import MSClsTrainer
    trainer=MSClsTrainer(work_dir="./work_dirs", config_name="r50_0.1_linear.yaml")
    print(trainer.cfg)
    trainer.train(validate=True)
   
### vit-small+Bigearthnet
   完全finetune：91.1%
  
    from aiearth.deeplearning.trainer.multispectral import MSClsTrainer
    trainer=MSClsTrainer(work_dir="./work_dirs", config_name="vit_small_1.0_finetune.yaml")
    print(trainer.cfg)
    trainer.train(validate=True)
   
   线性fientune：XXX
   
    from aiearth.deeplearning.trainer.multispectral import MSClsTrainer
    trainer=MSClsTrainer(work_dir="./work_dirs", config_name="vit_small_0.1_linear.yaml")
    print(trainer.cfg)
    trainer.train(validate=True)
    
### swin-tiny+Bigearthnet

完全finetune： 90.7%

    from aiearth.deeplearning.trainer.multispectral import MSClsTrainer
    trainer=MSClsTrainer(work_dir="./work_dirs", config_name="swin_tiny_1.0_finetune.yaml")
    print(trainer.cfg)
    trainer.train(validate=True)
    
线性finetune：xxx

    from aiearth.deeplearning.trainer.multispectral import MSClsTrainer
    trainer=MSClsTrainer(work_dir="./work_dirs", config_name="swin_tiny_0.1_linear.yaml")
    print(trainer.cfg)
    trainer.train(validate=True)

### 多卡训练
    
将上述语句写入一个xxx.py文件，然后如下调用：
   
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12301 test_ms.py
    
    
### 测试:
   
    from aiearth.deeplearning.trainer.multispectral import MSClsTrainer
    trainer=MSClsTrainer(work_dir="./work_dirs", config_name="r50.yaml")
    print(trainer.cfg)
    trainer.test("写入模型地址")
    
    


