#!/usr/bin/env python
"""nanosuite - Utilities to work with DNA nanotechnology
"""

from setuptools import setup

setup(name='Nanosuite',
      version='0.2.0',
      description='Utilities for work with DNA nanotechnology',
      author='Harold Fellermann',
      author_email='harold@nanovery.co.uk',
      url='http://harfel.teerun.de/',
      packages=['nanosuite'],
      install_requires=[
        "numpy",
        "openpyxl",
        "scipy",
        "lmfit",
      ]
)
