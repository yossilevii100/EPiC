from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from setuptools import setup, find_packages
import builtins

builtins.__ETW_PT_UTILS_SETUP__ = True
import etw_pytorch_utils

setup(
    name='etw_pytorch_utils',
    version=etw_pytorch_utils.__version__,
    author='Erik Wijmans',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'tqdm', 'visdom', 'future', 'statistics'])
