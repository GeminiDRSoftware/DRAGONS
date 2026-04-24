# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('../src/igrinsdr'))


# -- Project information -----------------------------------------------------

project = 'igrinsdr'
copyright = '2025, IGRINS2 team'
author = 'leejjoon'

# The full version, including alpha/beta/rc tags
try:
    import igrinsdr
    release = igrinsdr.__version__
except ImportError:
    release = "unknown"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# Ensure the package is in the Python path
import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath('../../')
sys.path.insert(0, project_root)

# Verify the package can be imported
try:
    import igrinsdr
    print(f"Successfully imported igrinsdr from: {igrinsdr.__file__}")
except ImportError as e:
    print(f"Error importing igrinsdr: {e}")
    print(f"Current sys.path: {sys.path}")

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = [('Returns', 'params_style')]

# Add extensions
extensions = [
    "sphinx_copybutton",
    # "sphinx_gallery.gen_gallery",
    "sphinx.ext.napoleon",
    "numpydoc",
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.inheritance_diagram',
    'myst_parser',
]

sphinx_gallery_conf = {
    # "examples_dirs": "../examples",  # path to your example scripts
    # "gallery_dirs": "examples",  # path to where to save gallery generated output
    "filename_pattern": "/.*",
    "ignore_pattern": "/_.*", # https://www.debuggex.com/
}


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Autosummary configuration
autosummary_generate = True
autosummary_imported_members = True
numpydoc_show_class_members = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# from DRAGONS

def setup(app):
   app.add_css_file('rtd_theme_overrides.css')
