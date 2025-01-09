#!/usr/bin/env python
"""nanosuite - Utilities to work with DNA nanotechnology
"""

from setuptools import setup    # type: ignore
from nanosuite import __version__

setup(name='Nanosuite',
      version=__version__,
      description='Utilities for work with DNA nanotechnology',
      author='Harold Fellermann',
      author_email='harold@nanovery.co.uk',
      url='http://harfel.teerun.de/',
      packages=['nanosuite'],
      include_package_data=True,
      install_requires=[
        "numpy",
        "openpyxl", # required by pandas.read_excel
        "pandas",
        "xarray",
        "scipy",
        "lmfit",
        "ply",
      ]
)
