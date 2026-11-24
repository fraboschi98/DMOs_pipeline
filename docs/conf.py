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
copyright = "2026, University of Luxembourg, LCSB, Digital Medicine group"
toc_object_entries_show_parents = "hide"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True

# gaitmap is heavy; mock it so autodoc builds without installing it.
autodoc_mock_imports = ["gaitmap", "gaitmap_mad"]

#html_theme = "sphinx_rtd_theme"
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "show_prev_next": False,
    "navbar_persistent": [],
    "navbar_end": ["search-field.html", "theme-switcher", "navbar-icon-links"],
    "show_nav_level": 2,
    "show_toc_level": 2,
    "footer_start": ["copyright"],
    "footer_end": [],
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
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"

#%% To add banner in lateral sidebar
templates_path = ["_templates"]

html_sidebars = {
    "index": [],
    "api": ["sidebar-image.html", "page-toc.html"],
    "user_guide/*": ["sidebar-image.html", "sidebar-nav-bs.html"],
    "examples/*": ["sidebar-image.html", "sidebar-nav-bs.html"],
    "**": ["sidebar-image.html", "page-toc.html"],
}

