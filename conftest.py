"""Make the project packages importable in tests without an editable install."""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(__file__))

# pyqsp/pyqsp deps emit noisy deprecation and solver-progress output.
warnings.filterwarnings("ignore")
