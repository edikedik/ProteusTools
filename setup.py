from distutils.core import setup

from setuptools import find_packages

setup(
    name='ProteusTools',
    version='0.1dev1',
    author='Ivan Reveguk',
    packages=find_packages(),
    install_requires=[
        'click>=7.1.2',
        'pandas>=1.1.3',
    ],
    scripts=[
        'affinity.py'
    ]
)
