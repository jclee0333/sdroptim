# coding: utf-8
'''Setup sdroptim package.'''
from codecs import open
from setuptools import setup, find_packages

with open('README.md', 'r', 'utf-8') as f:
    readme = f.read()

setup(
    name             = 'sdroptim',
    version          = '0.2.3',
    packages         = find_packages(),
    description      = 'Automated Machine Learning Library for KISTI Science Data Repository (for slurm job library)',
    long_description = readme,
    long_description_content_type = 'text/x-rst',
    license          = '',
    author           = 'Jeongcheol Lee',
    author_email     = 'jclee@kisti.re.kr',
    url              = 'https://github.com/jclee0333/sdroptim',
    #download_url     = 'Git에 저장된 whl 배포용 압축파일',
    install_requires = [
                        'featuretools', # using for feature synthesis and filtering
                        'nni', # using for feature selection tools (wrapper-based)
                        'optuna==2.0.0',
                        'mpi4py',
                        'sqlalchemy',
                        'sqlalchemy-utils',
                        'psycopg2-binary',
                        'astunparse',
                        'numpy',
                        'plotly',
                        'pandas',
                        'pyarrow', # added for featuretools serialization
                        'fastparquet',# added for featuretools serialization
                        'pickle5',
                        'gpuinfo' # added for automatic gpu usage check
                        ], 
    classifiers      = ['Programming Language :: Python :: 3.6',
                        'Intended Audience :: Korea Institute of Science and Technology Information',
                        'License :: MIT License']
    )