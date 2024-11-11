"""Nanosuite

Toolset for working with experimental DNA nanotechnology.
"""
__version__ = '0.2.4rc3'


try:
    from .mars import Assay
    from .crn import from_string as crn_from_string

except ImportError:
    # occurs during installation when setup.py imports the __version__ info form __init__
    pass
