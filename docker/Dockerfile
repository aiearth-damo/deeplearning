FROM nvidia/cuda:11.1.1-cudnn8-runtime-centos7

RUN yum install -y bzip2 openssl python3 gcc mesa-libGL centos-release-scl --disablerepo=cuda && \
    yum install -y devtoolset-7 && \
    yum clean all

SHELL ["scl", "enable", "devtoolset-7"]

ADD docker/condarc /root/.condarc
RUN curl -o Miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda.sh  -b -p /root/config/miniconda3 && \
    rm -rf /root/anaconda-ks.cfg /root/LICENSE.MIT /root/LICENSE.WTFPLv2 /root/README.md Miniconda.sh && \
    export PATH=/root/config/miniconda3/bin:$PATH && \
    conda install -y python=3.10
    #conda create -c https://conda.anaconda.org/conda-forge -n base python=3.7
    #conda install -y python=3.7 pip -c https://conda.anaconda.org/conda-forge

ENV PATH=/root/config/miniconda3/bin:$PATH

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pip -U --trusted-host pypi.tuna.tsinghua.edu.cn&& \
    pip config set global.index-url http://pypi.douban.com/simple/
    #pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple


ADD requirements.txt /root/requirements.txt

RUN pip install --trusted-host pypi.douban.com -r /root/requirements.txt && pip cache purge
RUN mim install mmcv-full -i http://pypi.doubanio.com/simple/ --trusted-host pypi.doubanio.com

ADD ./docker/runner /root/runner
ADD ./aietorch /root/aie-torch/aietorch
ADD ./setup.py /root/aie-torch/setup.py
RUN pip install -e /root/aie-torch/

ENTRYPOINT ["/root/runner/runner"]
