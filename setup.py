from distutils.core import setup
from pathlib import Path

from setuptools import find_packages

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name='ProteusTools',
    version='0.2dev1',
    author='Ivan Reveguk',
    author_email='ivan.reveguk@polytechnique.edu',
    description='Various workflows built around protMC',
    long_description=README,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Linux",
    ],
    python_requires='>=3.8',
    install_requires=[
        'ray>=1.2.0',
        'genetic @ git+https://github.com/skoblov-lab/genetic.git@0.2dev1',
        'networkx>=2.5.1',
        'click>=7.1.2',
        'pandas>=1.1.3',
        'tqdm>=4.50.2',
        'numpy>=1.19.1',
    ],
    package_data={
        'resources': [
            'protMC.exe',
            'ADAPT.conf',
            'MC.conf'
        ]
    },
    scripts=['ProteusTools/protmc/resources/protMC.exe'],
    packages=find_packages(),
)
