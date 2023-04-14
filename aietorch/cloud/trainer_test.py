import aie
from trainer import CloudModel



if __name__ == '__main__':
    # 获取aie样本集
    #aie.Authenticate("f273f8427261ebd774becb3a91c7966")
    aie.Authenticate()
    aie.g_var.set_var(aie.g_var.GVarKey.Log.LOG_LEVEL, aie.g_var.LogLevel.DEBUG_LEVEL)

    model_name = "sdk_test"
    category_json = [{"a": "n"}]
    img_std = [127,127,127]
    img_mean = [0,0,0]
    fp16 = False
    gpu_num = 2
    desc = "test"
    code_dir = "."

    model_id=CloudModel.model_create(code_dir, model_name=model_name, categoryJson=category_json, img_std=img_std,
                            img_mean=img_mean, fp16=fp16, gpu_num=gpu_num, desc=desc)
    print(model_id)
    #CloudModel.code_package_upload(model_id, "test.zip")
    #model_id = 563
    ret = CloudModel.model_info(model_id)
    #CloudModel.code_package_upload(model_id, "test.zip")
    print(ret)
