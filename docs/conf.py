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
  'sphinx.ext.doctest',
  'sphinx.ext.coverage',
  # 'sphinx.ext.mathjax',
  'sphinx.ext.viewcode',
  'sphinx.ext.autosummary',
  'IPython.sphinxext.ipython_console_highlighting',
  'IPython.sphinxext.ipython_directive',
  'sphinx.ext.intersphinx',
  'nbsphinx',
  'matplotlib.sphinxext.plot_directive',
  'sphinxcontrib.katex',
  'sphinx_sitemap',
  'sphinxcontrib.googleanalytics',
]
googleanalytics_id = 'G-H82NH09HYY'

html_baseurl = 'https://picmol.readthedocs.io/'
sitemap_url_scheme = "{link}"
sitemap_filename = 'sitemap2.xml' # readthedocs generates its own

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

html_theme = 'nature'
html_static_path = ['_static']

