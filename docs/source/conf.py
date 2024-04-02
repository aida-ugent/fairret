# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'fairret'
copyright = '2024, Maarten Buyl, MaryBeth Defrance'
author = 'Maarten Buyl, MaryBeth Defrance'
release = '0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

html_theme = "nature"

extensions = ['sphinx.ext.doctest', 'sphinx.ext.autodoc', 'sphinx.ext.autosummary', 'sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = []


# Napoleon settings
napoleon_google_docstring = True
napoleon_include_init_with_doc = True

# Autodoc settings
autodoc_member_order = 'bysource'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# -- Set system path in order to import code ---------------------------------
import pathlib
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

print(pathlib.Path(__file__).parents[2].resolve().as_posix())
import fairret