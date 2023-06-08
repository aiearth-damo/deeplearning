import os
import json
import urllib.parse
import requests
from tempfile import gettempdir, NamedTemporaryFile
import aie
from aie.client import BaseClient, Endpoints
from aie.error import AIEError, AIEErrorCode
from aiearth.deeplearning.utils.file import zip_dir
import aie.g_var as g_var

class AIECloudException(Exception):
    def __init__(self, aie_resp_json) -> None:
        self.error_code = aie_resp_json["code"]
        self.message = aie_resp_json["message"]
        super().__init__(self.message)


class AIEClient(BaseClient):

    @staticmethod
    def __append_extra_hdrs(hdrs):
        hdrs["x-aie-auth-token"] = aie.auth.Authenticate.getCurrentUserToken()
        return hdrs

    @staticmethod
    def handle_resp(resp):
        if resp.status_code != 200:
            if "401 Authorization Required" in resp.text:
                raise AIEError(AIEErrorCode.ENVIRONMENT_INIT_ERROR,
                               f"未授权或者个人token失效，请先调用 aie.Authenticate() 进行授权")
            else:
                raise AIEError(AIEErrorCode.DEFAULT_INTERNAL_ERROR,
                               "", f"http请求错误: {resp.text}")

        if aie.aie_env.AIEEnv.getDebugLevel() == g_var.LogLevel.DEBUG_LEVEL:
            print(
                f"BaseClient::post response. url: {resp.url}, response: {resp.json()}")

        return resp
    
    @staticmethod
    def log(message):
        if aie.aie_env.AIEEnv.getDebugLevel() == g_var.LogLevel.DEBUG_LEVEL:
            print(message)

    # support submit files
    @staticmethod
    def post(url, hdrs, data, files=None, append_extra_hdrs=True, use_json=True):
        if append_extra_hdrs:
            hdrs = AIEClient.__append_extra_hdrs(hdrs)

        AIEClient.log(f"BaseClient::post request. url: {url}, headers: {json.dumps(hdrs)}, data: {json.dumps(data)}")

        if use_json:
            resp = requests.post(url=url, headers=hdrs, timeout=(600, 600),
                             json=data, files=files, verify=False)
        else:
            resp = requests.post(url=url, headers=hdrs, timeout=(600, 600),
                             data=data, files=files, verify=False)
        return AIEClient.handle_resp(resp)
    
    # support submit files
    @staticmethod
    def put(url, hdrs, data, files=None, append_extra_hdrs=True, use_json=True):
        if append_extra_hdrs:
            hdrs = BaseClient.__append_extra_hdrs(hdrs)

        AIEClient.log(f"BaseClient::put request. url: {url}, headers: {json.dumps(hdrs)}, data: {json.dumps(data)}")

        resp = requests.put(url=url, headers=hdrs, timeout=(600, 600),
                             json=data, verify=False)
        return AIEClient.handle_resp(resp)
    
    @staticmethod
    def get(url, data, hdrs={}, append_extra_hdrs=True):
        url = url + "?" + urllib.parse.urlencode(data)
        return BaseClient.get(url, hdrs, append_extra_hdrs)


        
