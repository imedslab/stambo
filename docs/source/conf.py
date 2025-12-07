# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import sys
import os
import furo
from stambo import __version__

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../notebooks'))

project = 'stambo'
copyright = '2024-now, Aleksei Tiulpin'
author = 'Aleksei Tiulpin'
version = __version__
release = f"v{version}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc.typehints",  # <— Ensures type hint formatting
    "nbsphinx",
    "nbsphinx_link",
]

# Make Sphinx turn type hints into cross-references
autodoc_typehints = "both"  # show in signature + show in description
autodoc_typehints_format = "short"  # use short names (Path not pathlib.Path)

# If you want type hints to be pulled *from docstrings*:
autodoc_typehints_description_target = "documented"  # or "all"

# Intersphinx so built-in and external types become links
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"

