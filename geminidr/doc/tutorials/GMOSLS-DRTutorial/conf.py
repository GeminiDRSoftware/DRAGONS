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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Tutorial Series - GMOS Longslit Data Reduction with DRAGONS'
copyright = '2024, Association of Universities for Research in Astronomy'
author = 'DRAGONS Team'

# The full version, including alpha/beta/rc tags
version = '4.0-dev'
#release = '3.2.2'
#rtdurl = 'v'+release
#release = '3.2.x'
#rtdurl = 'release-'+release
rtdurl = 'latest'

today = '2025'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.imgmath'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = 'TutorialSeries-GMOSLSDRDRAGONS'

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = None
#html_logo = "_graphics/DRAGONS - Icon blue.png"


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'TutorialSeries-GMOSLSDRDRAGONS.tex', 'Tutorial Series - GMOS Longslit Data Reduction with DRAGONS',
     'DRAGONS Team', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'tutorialseries-gmoslsdrdragons', 'Tutorial Series - GMOS Longslit Data Reduction with DRAGONS',
     [author], 1)
]


# --- Customizations ----------------


def setup(app):
    app.add_css_file('todo-styles.css')
    app.add_css_file('rtf_theme_overrides.css')
    app.add_css_file('css/rtd_theme_overrides_references.css')
    app.add_css_file('fonts.css')


rst_epilog = """
.. role:: raw-html(raw)
   :format: html

.. |caldb| replace:: :raw-html:`<a href="https://dragons-recipe-system-users-manual.readthedocs.io/en/{v}/caldb.html" target="_blank">caldb</a>`
.. |dataselect| replace:: :raw-html:`<a href="https://dragons-recipe-system-users-manual.readthedocs.io/en/{v}/supptools.html#dataselect" target="_blank">dataselect</a>`
.. |descriptors| replace:: :raw-html:`<a href="https://astrodata.readthedocs.io/en/{v}/appendix_descriptors.html" target="_blank">descriptors</a>`
.. |descriptor| replace:: :raw-html:`<a href="https://astrodata.readthedocs.io/en/{v}/appendix_descriptors.html" target="_blank">descriptors</a>`
.. |reduce| replace:: :raw-html:`<a href="https://dragons-recipe-system-users-manual.readthedocs.io/en/{v}/reduce.html" target="_blank">reduce</a>`
.. |showd| replace:: :raw-html:`<a href="https://dragons-recipe-system-users-manual.readthedocs.io/en/{v}/supptools.html#showd" target"_blank">showd</a>`
.. |showrecipes| replace:: :raw-html:`<a href="https://dragons-recipe-system-users-manual.readthedocs.io/en/{v}/supptools.html#showrecipes" target="_blank">showrecipes</a>`
.. |showpars| replace:: :raw-html:`<a href="https://dragons-recipe-system-users-manual.readthedocs.io/en/{v}/supptools.html#showpars" target="_blank">showpars</a>`
.. |typewalk| replace:: :raw-html:`<a href="https://dragons-recipe-system-users-manual.readthedocs.io/en/{v}/supptools.html#typewalk" target="_blank">typewalk</a>`
.. |atfile| replace:: :raw-html:`<a href="https://dragons-recipe-system-users-manual.readthedocs.io/en/{v}/reduce.html#the-file-facility" target="_blank">"at-file" Facility</a>`
.. |astrodatauser| replace:: :raw-html:`<a href="https://astrodata.readthedocs.io/en/{v}/usermanual/index.html" target="_blank">Astrodata User Manual</a>`

.. |GMOSImgTut| replace:: :raw-html:`<a href="http://GMOSImg-DRTutorial.readthedocs.io/en/{v}">GMOS Imaging Data Reduction Tutorial</a>`
.. |DRAGONS| replace:: :raw-html:`<a href="https://dragons.readthedocs.io/en/{v}/" target="_blank">DRAGONS</a>`

.. |RSUserInstall| replace:: :raw-html:`<a href="https://dragons.readthedocs.io/projects/recipe-system-users-manual/en/{v}/install.html" target="_blank">DRAGONS Installation Instructions</a>`
""".format(v = rtdurl)
