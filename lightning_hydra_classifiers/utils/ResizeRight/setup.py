

# from distutils.core import setup #, find_packages
from setuptools import setup, find_packages
import sys

# print(sys.path)
# print(dir())

setup(
    name='ResizeRight',
    version='0.1dev',
    packages=find_packages(include=['ResizeRight', 'resize_right', 'resize_right.*']),
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
)
