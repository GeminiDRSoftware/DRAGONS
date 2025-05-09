#
# Tutorial Series - NIRI Imaging Data Reduction with DRAGONS documentation build configuration file, created by
# sphinx-quickstart on Mon Aug 13 15:54:35 2018.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.imgmath']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'Tutorial Series - NIRI Imaging Data Reduction with DRAGONS'
copyright = '2025, Association of Universities for Research in Astronomy'
author = 'DRAGONS Team'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = '4.1_dev'
# The full version, including alpha/beta/rc tags.
#release = '4.1.0'
#rtdurl = 'v'+release
release = '4.1.x'
rtdurl = 'release-'+release
#rtdurl = 'latest'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
today = 'September 2025'
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
# html_sidebars = {
#    '**': [
#        'about.html',
#        'navigation.html',
#        'relations.html',  # needs 'show_related': True theme option to display
#        'searchbox.html',
#        'donate.html',
#    ]
# }


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'TutorialSeries-NIRIImgDRDRAGONS'


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
    (master_doc, 'TutorialSeries-NIRIImgDRDRAGONS.tex', 'Tutorial Series - NIRI Imaging Data Reduction with DRAGONS',
     'DRAGONS Team', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'tutorialseries-niriimgdrdragons', 'Tutorial Series - NIRI Imaging Data Reduction with DRAGONS',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'TutorialSeries-NIRIImgDRDRAGONS', 'Tutorial Series - NIRI Imaging Data Reduction with DRAGONS Documentation',
     author, 'TutorialSeries-NIRIImagingDataReductionwithDRAGONS', 'One line description of project.',
     'Miscellaneous'),
]



# -- Options for Epub output ----------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']



# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'python': ('http://docs.python.org/3', None)}

# ------

# Adding style in order to have the todos show up in a red box


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

.. |DRAGONS| replace:: :raw-html:`<a href="https://dragons.readthedocs.io/en/{v}/" target="_blank">DRAGONS</a>`

.. |RSUserInstall| replace:: :raw-html:`<a href="https://dragons.readthedocs.io/projects/recipe-system-users-manual/en/{v}/install.html" target="_blank">DRAGONS Installation Instructions</a>`
""".format(v = rtdurl)
