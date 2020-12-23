from distutils.core import setup

from setuptools import find_packages

setup(
    name='ProteusTools',
    version='0.1dev2',
    author='Ivan Reveguk',
    packages=find_packages(),
    install_requires=[
        'click>=7.1.2',
        'pandas>=1.1.3',
        'tqdm>=4.50.2',
        'seaborn>=0.11.0',
        'multiprocess>=0.70.10',
        'numpy>=1.19.1',
        'biotite>=0.24'
    ],
)
