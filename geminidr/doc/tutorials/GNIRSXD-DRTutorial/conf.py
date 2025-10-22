# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Tutorial Series - GNIRS Cross-dispersed Data Reduction with DRAGONS'
copyright = '2025, DRAGONS Team'
author = 'DRAGONS Team'

# The full version, including alpha/beta/rc tags
version = '4.1'
#release = '4.1.0'
#rtdurl = 'v'+release
release = '4.1.x'
rtdurl = 'release-'+release
#rtdurl = 'latest'

today = 'October 2025'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.imgmath',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
exclude_patterns.append('notused')

source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = 'TutorialSeries-GNIRSXDDRDRAGONS'

# Output file base name for HTML help builder.
htmlhelp_basename = 'TutorialSeries-GNIRSLSDRDRAGONS'

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
    (master_doc, 'TutorialSeries-GMOSXDDRDRAGONS.tex', 'Tutorial Series - GMOS Cross-dispersed Data Reduction with DRAGONS Documentation',
     'DRAGONS Team', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'tutorialseries-gnirsxddrdragons', 'Tutorial Series - GNIRS Cross-dispersed Data Reduction with DRAGONS Documentation',
     [author], 1)
]

# --- Customizations ----------------

def setup(app):
    app.add_css_file('todo-styles.css')
    app.add_css_file('rtf_theme_overrides.css')
    app.add_css_file('rtd_theme_overrides_references.css')
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

.. |GNIRSImgTut| replace:: :raw-html:`<a href="http://GNIRSImg-DRTutorial.readthedocs.io/en/{v}">GNIRS Imaging Data Reduction Tutorial</a>`
.. |GNIRSLSTut| replace:: :raw-html:`<a href="http://GNIRSLS-DRTutorial.readthedocs.io/en/{v}">GNIRS Longslit Data Reduction Tutorial</a>`
.. |DRAGONS| replace:: :raw-html:`<a href="https://dragons.readthedocs.io/en/{v}/" target="_blank">DRAGONS</a>`

.. |RSUserInstall| replace:: :raw-html:`<a href="https://dragons.readthedocs.io/projects/recipe-system-users-manual/en/{v}/install.html" target="_blank">DRAGONS Installation Instructions</a>`
""".format(v = rtdurl)
