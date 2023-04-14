import os
os.environ['SDK_CLIENT_HOST'] = 'https://pre-engine-aiearth.aliyun.com'

import aie
import aie.g_var as g_var


from aietorch.datasets.aie.aie_dataset import AIEDataSet

aie.Authenticate("7e201c7da16407605c00f7c433d78baf")
#aie.Initialize()
g_var.set_var(g_var.GVarKey.Log.LOG_LEVEL, aie.g_var.LogLevel.INFO_LEVEL)

#样本集ID=262 单景
myDataSet = AIEDataSet(262)
info = myDataSet.to_mmseg_datasets()
print(info)

#样本集ID=276 双景
myDataSet2 = AIEDataSet(276)
info = myDataSet2.to_mmseg_datasets()
print(info)

#样本集ID=158 纯影像数据集
myDataSet3 = AIEDataSet(158)
info = myDataSet3.to_mmseg_datasets()
print(info)