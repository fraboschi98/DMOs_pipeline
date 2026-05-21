# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:39:26 2026

@author: francesca.boschi
"""

import sys
from pathlib import Path

# Add project root to Python path so Sphinx can import dmos_pipeline
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "DMOs Pipeline"
author = "Francesca Boschi"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True

html_theme = "sphinx_rtd_theme"