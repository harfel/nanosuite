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
      url='https://github.com/harfel/nanosuite',
      license_files = ('LICENSE.txt',),
      packages=['nanosuite'],
      include_package_data=True,
      install_requires=[
        "diffrax",
        "jax",
        "lmfit",
        "openpyxl", # required by pandas.read_excel
        "pandas",
        "ply",
        "scipy",
        "xarray",
        "git+https://github.com/google-deepmind/xarray_jax.git@v0.1.0",
      ]
)
