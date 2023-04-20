import os

from aie.client import BaseClient, Endpoints
from aie.error import AIEError
import aie
import aie.g_var as g_var

class TestDataSet(BaseClient):
    __sampleinfo_host = Endpoints.HOST + "/trainSDK"
    __sampleinfo_api_path = "/api/sdk/sampleset/dinfo?id="
    __samplelist_api_path = "/api/sdk/sampleset/list"
    __categorylist_api_path = "/api/sdk/category/list"

    def __get_sampleset__(self, dataset_id):
        print("requelt sampletset info start")
        headers = {"Content-Type": "application/json"}
        get_sampleset_url = self.__sampleinfo_host + self.__sampleinfo_api_path + str(dataset_id)
        env = os.getenv(g_var.JupyterEnv.TOKEN)
        if env is None:
            get_sampleset_url = '{}&env={}'.format(get_sampleset_url, 'local')
        else:
            get_sampleset_url = '{}&env={}'.format(get_sampleset_url, 'cloud')
        print(get_sampleset_url)
        response = super(TestDataSet, self).get(
            get_sampleset_url, headers).json()
        print(response)
        if response["code"] != 0:
            raise AIEError(response["code"], response["message"])
        print("requelt sampletset info end")

    def __get_sampleset_list__(self):
        print("requelt sampletset list info start")
        headers = {"Content-Type": "application/json"}
        paramsJson = {
            "name":"GID", #样本集名称,可以为空
            "nameMatchRule":"like", #样本集名称匹配规则，可以为空, 默认为全匹配; like:模糊匹配, equal:全匹配
            "cropSize":"512*512", #样本剪裁大小，可以为空; 目前支持:512*512、1024*1024、2048*2048
            "categoryMd5":"88de432dcd12650bc6e10f20ad82193e", #类目标识, 可以为空
            "type": 2, #样本集类型, 可以为空, 目前支持: 1:自定义样本集, 2:公开样本集
            "pageNo":1,  #分页页数, 可以为空, 默认为1
            "pageSize":5  #分页每页个数, 可以为空, 默认为10
        }
        post_sampleset_url = self.__sampleinfo_host + self.__samplelist_api_path
        print(post_sampleset_url)
        response = super(TestDataSet, self).post(
            post_sampleset_url, headers, paramsJson).json()
        print(response)
        if response["code"] != 0:
            raise AIEError(response["code"], response["message"])
        print("requelt sampletset list info end")

    def __get_category_list__(self):
        print("requelt category list info start")
        headers = {"Content-Type": "application/json"}
        paramsJson = {
            "name": "预置", #标签名称,可以为空
            "nameMatchRule":"like", #标签名称匹配规则，可以为空, 默认为全匹配; like:模糊匹配, equal:全匹配
            "pageNo":1,  #分页页数, 可以为空, 默认为1
            "pageSize":5  #分页每页个数, 可以为空, 默认为10
        }
        post_sampleset_url = self.__sampleinfo_host + self.__categorylist_api_path
        print(post_sampleset_url)
        response = super(TestDataSet, self).post(
            post_sampleset_url, headers, paramsJson).json()
        print(response)
        if response["code"] != 0:
            raise AIEError(response["code"], response["message"])
        print("requelt category list info end")

#Test
aie.Authenticate("1abba48018701cfe836bd07f291778b3")


g_var.set_var(g_var.GVarKey.Log.LOG_LEVEL, aie.g_var.LogLevel.INFO_LEVEL)
myDataSet = TestDataSet()
#下载样本集
#myDataSet.__get_sampleset__(600039)
#查询样本集列表
myDataSet.__get_sampleset_list__()
#查询类目列表
myDataSet.__get_category_list__()

