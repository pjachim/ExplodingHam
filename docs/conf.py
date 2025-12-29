"""Sphinx configuration for ExplodingHam documentation."""

import os
import sys

# Add the source directory to the path so autodoc can find the modules
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'ExplodingHam'
copyright = '2025, Peter Jachim'
author = 'Peter Jachim'

# The short X.Y version
version = '0.0.5'
# The full version, including alpha/beta/rc tags
release = '0.0.5'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings
extensions = [
    'sphinx.ext.autodoc',      # Automatic documentation from docstrings
    'sphinx.ext.napoleon',     # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',     # Add links to highlighted source code
    'sphinx.ext.intersphinx',  # Link to other project's documentation
    'sphinx.ext.mathjax',      # Render math via JavaScript
    'myst_parser',             # Support for Markdown files
]

# Add any paths that contain templates here, relative to this directory
templates_path = ['_templates']

# The suffix(es) of source filenames
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
}

# Add any paths that contain custom static files (such as style sheets)
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names
html_sidebars = {
    '**': [
        'globaltoc.html',
        'relations.html',
        'sourcelink.html',
        'searchbox.html',
    ]
}

# -- Options for autodoc -----------------------------------------------------

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method
autodoc_typehints = 'description'

# Don't show class signature with the class' name
autodoc_class_signature = 'separated'

# -- Options for intersphinx -------------------------------------------------

# Example configuration for intersphinx: refer to the Python standard library
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# -- Options for Napoleon (NumPy-style docstrings) ---------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for MyST --------------------------------------------------------

myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'dollarmath',
    'fieldlist',
    'html_admonition',
    'html_image',
    'replacements',
    'smartquotes',
    'strikethrough',
    'substitution',
    'tasklist',
]

myst_heading_anchors = 3
