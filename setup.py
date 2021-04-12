from distutils.core import setup

from setuptools import find_packages

setup(
    name='ProteusTools',
    version='0.1dev3',
    author='Ivan Reveguk',
    author_email='ivan.reveguk@polytechnique.edu',
    description='A (growing) collection of high-level interface tools handling protmc',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Linux",
    ],
    python_requires='>=3.8',
    packages=find_packages(),
    # TODO: verify packages' versions
    install_requires=[
        'pytest',
        'ray',
        'genetic',
        'networkx',
        'click>=7.1.2',
        'pandas>=1.1.3',
        'tqdm>=4.50.2',
        'seaborn>=0.11.0',
        'multiprocess>=0.70.10',
        'numpy>=1.19.1',
        'biotite>=0.24'
    ],
)
