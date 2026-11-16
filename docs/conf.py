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

# gaitmap is heavy and may fail to install on Read the Docs: mock it so
# autodoc can build the API pages from docstrings without importing it.
autodoc_mock_imports = ["gaitmap", "gaitmap_mad"]

#html_theme = "sphinx_rtd_theme"
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "show_nav_level": 2,
    "show_toc_level": 1,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/fraboschi98/DMOs_pipeline",
            "icon": "fa-brands fa-github",
        },
        
    # TODO: add link to paper when it's published!
    # {
    #     "name": "Paper",
    #     "url": "https://doi.org/...",
    #     "icon": "fa-solid fa-file-lines",
    # },
    ],
}

#%% To make logo and favicon appear
html_static_path = ["_static"]
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"

#%% To add banner in lateral sidebar
templates_path = ["_templates"]

html_sidebars = {
    "**": [
        "sidebar-image.html",
        "sidebar-nav-bs.html",
    ]
}