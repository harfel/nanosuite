"""Unit tests for jupyter

Ensure that all tests are passing with the monkey patches from
the jupyter module activated.
"""
from unittest.mock import MagicMock

import nanosuite.jupyter
nanosuite.jupyter.display = MagicMock()

# Run tests for all modules that are monkey-patched in nanosuite.jupyter
from .test_equilibrium import *
from .test_mars import *
from .test_trajectory import *
