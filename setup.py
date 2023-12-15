#!/usr/bin/env python
"""nanosuite - Utilities to work with DNA nanotechnology
"""

from setuptools import setup
from nanosuite import __version__

setup(name='Nanosuite',
      version=__version__,
      description='Utilities for work with DNA nanotechnology',
      author='Harold Fellermann',
      author_email='harold@nanovery.co.uk',
      url='http://harfel.teerun.de/',
      packages=['nanosuite'],
      install_requires=[
        "numpy",
        "openpyxl",
        "pandas",
        "xarray",
        "scipy",
        "lmfit",
      ]
)
