from setuptools import setup, find_packages, find_namespace_packages
from setuptools.command.install import install as _install
from setuptools.command.egg_info import egg_info as _egg_info
import sys, os
import io
import os.path as op
import atexit

VERSION = "0.0.3"


# get the dependencies and installs
with io.open(op.join('requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]

class install(_install):
    def __init__(self, *args, **kwargs):
        super(install, self).__init__(*args, **kwargs)
        atexit.register(_post_install)


class egg_info(_egg_info):
    def __init__(self, *args, **kwargs):
        super(egg_info, self).__init__(*args, **kwargs)
        atexit.register(_post_egg_info)


def get_package_version(package_name):
    import re
    rule = package_name + "\W+"
    regex = re.compile(rule)
    for dep in install_requires:
        if dep == package_name or len(regex.findall(dep)) > 0:
            return dep


def install_pkg_from_install_requires(pkg_name, cmd_prefix="pip install"):
    pkg = get_package_version(pkg_name)
    print("pkgname", pkg)
    if pkg:
        from subprocess import run
        cmd = "%s '%s'" %(cmd_prefix, pkg)
        print(cmd)
        run(cmd, shell=True, capture_output=False)


def _post_install():
    print("post install task")
    from subprocess import run
    cmd = "mim install 'mmcv-full<=1.7.1'"
    run(cmd, shell=True, capture_output=False)


def _post_egg_info():
    print("post egg info task")
    from subprocess import run
    install_pkg_from_install_requires("openmim")
    install_pkg_from_install_requires("torch")


packages = find_namespace_packages(include=['aiearth.*'])
setup(name='aiearth-deeplearning',
      version=VERSION,
      description='AI Earth algo sdk builded by torch && mmseg',
      author='AI Earth developer team',
      author_email='yuanbin.myb@alibaba-inc.com',
      packages=packages,
      python_requires='>=3.7.2',
      include_package_data=True,
      cmdclass={
        'install': install,
        'egg_info': egg_info,
      },
      install_requires=install_requires,
)
