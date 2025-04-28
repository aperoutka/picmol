# Configuration file for the Sphinx documentation builder.
import os, sys
sys.path.insert(0, os.path.abspath('../')) 


# -- Project information -----------------------------------------------------

project = 'picmol'
copyright = '2025, Allison A. Peroutka'
author = 'Allison A. Peroutka'
release = '0.0.1'


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'
html_static_path = ['_static']

