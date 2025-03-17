"""Nanosuite

Toolset for working with experimental DNA nanotechnology.
"""
__version__ = '0.3.0rc2'


def _is_notebook() -> bool:
    """Detect if code is run in a jupyter notebook"""
    try:
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        if shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


if _is_notebook():
    from . import jupyter

try:
    from .mars import Assay
    from .crn import from_string as crn_from_string

except ImportError:
    # occurs during installation when setup.py imports the __version__ info form __init__
    pass
