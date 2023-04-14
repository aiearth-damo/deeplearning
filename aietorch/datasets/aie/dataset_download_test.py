import requests
import shutil
import tempfile
import os
import uuid

file_url = "https://aiearth-cd.oss-cn-shanghai.aliyuncs.com/data/fanymkhmokuyoday.zip?OSSAccessKeyId=LTAI5tCYp3owUbkXm19N8mxK&Expires=361675070302&Signature=9oQduuDFxWt6FA1jCbP%2FSoNlZlY%3D"
download_dir = "fanymkhmokuyoday/"
download_path = os.path.join(download_dir, "dataset")
if(not os.path.exists(download_path)):
    os.makedirs(download_path)
res = requests.get(url=file_url)
_tmp_file = tempfile.NamedTemporaryFile()
_tmp_file.write(res.content)
shutil.unpack_archive(_tmp_file.name, download_path, format="zip")
print(download_path + os.listdir(download_path)[0] + "/")
os.path.sep