import math
import re
import requests
from aiearth.deeplearning.cloud.aie_client import AIEClient, Endpoints
import aie
import os
from tqdm import tqdm

class PretrainedModel():
    api_group = "/trainSDK/api/model"
    host = Endpoints.HOST
    default_download_dir = os.path.join(os.path.expanduser("~"), ".cache", "aie")
    scheme = "aie://"


    def __init__(self, uri=None, model_type=None, model_name=None, model_format="pth", local_path=None):
        if uri != None:
            self.parse_uri(uri)
        else:
            if model_type == None or model_name == None:
                raise Exception("uri and model_type, model_name cannot both be None")
            self.model_type = model_type
            self.model_name = model_name
        self.model_format = model_format
        self.local_path = local_path if local_path != None else self.get_local_path()
        if not os.path.exists(self.local_path):
            self.download()

    def parse_uri(self, uri:str):
        assert uri.startswith(self.scheme)
        m = re.match(r"(?P<scheme>\w+)://(?P<model_type>\w+)/(?P<model_name>.+)", uri)
        if not m:
            raise Exception("uri %s is invalid" % (uri))
        self.model_type = m.group('model_type')
        self.model_name = m.group('model_name')


    def get_local_path(self):
        local_path = os.path.join(self.default_download_dir, self.model_type, self.model_name)
        return local_path

    def __info(self):
        api_path = "/pretrained"
        url = self.host + self.api_group + api_path
        data = {
            "modelName": self.model_name,
            "appType": self.model_type
        }
        ret = AIEClient.get(url, data)
        self.model_info = ret.json()['data']
        return self.model_info
    
    def download(self):
        model_info = self.__info()
        download_link = model_info['downloadLink']
        print("download start")
        # create download basedir
        base_dir = os.path.dirname(self.local_path)
        os.makedirs(base_dir, exist_ok=True)

        response_data_file = requests.get(url=download_link, stream=True)
        if response_data_file.status_code != 200:
            raise Exception("Download failed, resp code " + str(response_data_file.status_code))
        file_size = float(response_data_file.headers['Content-Length'])
        chunk_size = 1024 * 1024
        chunk_cnt = math.ceil(file_size/ chunk_size)
        with open(self.local_path, "wb") as f:
            with tqdm(total=chunk_cnt, unit="MiB") as bar:
                for chunk in response_data_file.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        bar.update(1)
        print("download end")



if __name__ == '__main__':
    aie.Authenticate()
    aie.g_var.set_var(aie.g_var.GVarKey.Log.LOG_LEVEL, aie.g_var.LogLevel.INFO_LEVEL)

    model = PretrainedModel("ChangeDet", "changedet_hrnet_w18_base_150k_new512_cosine_lr_batch_48_v25_finetune.pth")
    print(model.local_path)
    print(hasattr(model, "local_path"))
