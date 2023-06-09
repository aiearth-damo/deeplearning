# 定制化算法开发

根据任务类型选择不同的训练引擎(train.engine)。我们会在SDK中集成了多种广受社区使用的框架比如MMSegmentation, MMDetection, Detectron2等。如果你有相关框架的开发经验，即可快速开始进行你的模型定制。

下面我们以分割类任务使用MMSeg为例，注册一个用户自定义模型：

```Python
# use custom model
from aiearth.deeplearning.models.changedet import ChangedetEncoderDecoder
from aiearth.deeplearning.engine.mmseg.models.builder import SEGMENTORS

@SEGMENTORS.register_module
class MyChangedetEncoder(ChangedetEncoderDecoder):
    pass

trainer.cfg.model.type = "MyChangedetEncoder"
```

更多类型定制可以参考相关引擎的文档：
[MMSeg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/zh_cn/tutorials/customize_models.md)
