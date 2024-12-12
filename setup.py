#!/usr/bin/env python3
from setuptools import setup

long_description = "Package to interface with module-based (modified) suite2p by Georgios Keliris."

setup(
    name='gks2p',
    version='0.1.1',
    description="Interface with module-based suite2p.",
    long_description=long_description,
    author='Georgios A. Keliris',
    author_email='georgios.keliris@outlook.com',
    license='MIT',
    url='https://github.com/gkeliris/gks2p',
    keywords='suite2p module-based gks2p',
    packages=['gks2p'],
    install_requires=['tifffile', 'natsort'],
    classifiers=[
        'Development Status :: 1 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English'
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
