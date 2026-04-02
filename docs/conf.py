# Configuration file for the Sphinx documentation builder.
from __future__ import annotations

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
# Add project root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

# -- Project information -----------------------------------------------------
project = "opticalib"
author = "opticalib contributors"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

# The full version, including alpha/beta/rc tags
try:
    from opticalib.__version__ import __version__ as release
except Exception:
    # Fallback if package isn't installed/importable during doc build
    release = "0.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
    "myst_parser",
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_attr_annotations = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "show-inheritance": True,
    "inherited-members": True,
}

autodoc_typehints = "description"
typehints_use_signature = True
typehints_fully_qualified = False

# Avoid import-time failures for hardware/vendor deps
autodoc_mock_imports = [
    # hardware APIs that may not exist in doc build envs
    "opticalib.devices._API.alpaoAPI",
    "opticalib.devices._API.micAPI",
    "opticalib.devices._API.splattAPI",
    "opticalib.devices._API.i4d",
    # optional external package used by some dmutils modules
    "m4",
    # thin-plate-spline library used by pupil_calibration
    "tps",
]

# Intersphinx mappings for cross-references
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# Source suffixes for Sphinx to parse
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = f"{project} {release}"

# -- Options for TODOs -------------------------------------------------------
todo_include_todos = True
