#!/usr/bin/env python
# encoding: utf-8
# On Windows, you need to execute:
# set VS90COMNTOOLS=%VS100COMNTOOLS%
# python setup.py build_ext --compiler=msvc
from setuptools import setup
from sys import version_info as python_version
from os import path
from distutils.extension import Extension
from Cython.Distutils import build_ext

STATIC = False

install_requires = []
if python_version < (2, 7):
    new_27 = ['ordereddict', 'argparse']
    install_requires.extend(new_27)


ext_modules = []

# pykaldi library compilation (static|dynamic)
if STATIC:
    # STATIC TODO extract linking parameters from Makefile
    library_dirs, libraries = [], []
    extra_objects = ['pykaldi.a', ]
else:
    # DYNAMIC
    library_dirs = ['kaldi', ]
    libraries = ['pykaldi', ]
    extra_objects = []
ext_modules.append(Extension('kaldi.decoders',
                             language='c++',
                             include_dirs=['..', 'pyfst', ],
                             library_dirs=library_dirs,
                             libraries=libraries,
                             extra_objects=extra_objects,
                             sources=['kaldi/decoders.pyx', ],
                             ))


long_description = open(path.join(path.dirname(__file__), 'README.txt')).read()

try:
    # In order to find out the pykaldi version from installed package at runtime use:
    # import pgk_resources as pkg; pkg.get_distribution('pykaldi')
    from subprocess import check_output
    git_version = check_output(['git', 'rev-parse', 'HEAD'])
except:
    git_version = 'Unknown Git version'
    print(git_version)

setup(
    name='pykaldi',
    packages=['kaldi', ],
    package_data={'kaldi': ['libpykaldi.so', 'test_shortest.txt']},
    include_package_data=True,
    cmdclass={'build_ext': build_ext},
    version='0.1-' + git_version,
    install_requires=install_requires,
    setup_requires=['cython>=0.19.1', 'nose>=1.0'],
    ext_modules=ext_modules,
    test_suite="nose.collector",
    tests_require=['nose>=1.0', 'pykaldi'],
    author='Ondrej Platek',
    author_email='ondrej.platek@seznam.cz',
    url='https://github.com/DSG-UFAL/pykaldi',
    license='Apache, Version 2.0',
    keywords='Kaldi speech recognition Python bindings',
    description='C++/Python wrapper for Kaldi decoders',
    long_description=long_description,
    classifiers='''
        Programming Language :: Python :: 2
        License :: OSI Approved :: Apache License, Version 2
        Operating System :: POSIX :: Linux
        Intended Audiance :: Speech Recognition scientist
        Intended Audiance :: Students
        Environment :: Console
        '''.strip().splitlines(),
)
