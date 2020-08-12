# -*- coding: utf-8 -*-

"""Top-level package for bokeh-catplot."""

# Force showing deprecation warnings.
import warnings

from .cat import *
from .dist import *

warnings.filterwarnings("once", category=DeprecationWarning)

warnings.warn("bokeh-catplot is deprecated. Use iqplot instead.", DeprecationWarning)

__author__ = """Justin Bois"""
__email__ = "bois@caltech.edu"
__version__ = "0.1.9"
