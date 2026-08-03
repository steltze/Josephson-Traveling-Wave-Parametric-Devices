"""Sphinx configuration for the Traveling Wave Parametric Devices docs."""

from __future__ import annotations

import os
import sys

# `src` (not the repo root) is the import root: the code imports as
# `from models import ...`, `from analysis.checks import ...`, etc. -
# see `pythonpath = ["src"]` in pyproject.toml's [tool.pytest.ini_options].
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------

project = "Traveling Wave Parametric Devices"
copyright_holder = "Stelios Tzelepis"
author = copyright_holder
copyright = f"2026, {copyright_holder}"
release = "0.1.0"

# -- General configuration -----------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Public classes are re-exported in package __init__.py files (e.g. both
# `simulation.SimulationConfig` and `simulation.config.SimulationConfig`
# resolve), so bare-name type cross-references in numpy-style docstrings
# are inherently ambiguous between the re-export and the definition site.
# Both candidates are valid; suppress the warning rather than the xrefs.
suppress_warnings = ["ref.python"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Autodoc / autosummary ------------------------------------------------

autosummary_generate = True
autosummary_imported_members = False
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = False

# -- Intersphinx -----------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
}

# -- MyST (Markdown) --------------------------------------------------------

myst_enable_extensions = ["dollarmath", "colon_fence"]

# -- HTML output -------------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_title = project

html_theme_options = {
    "sidebar_hide_name": False,
}
