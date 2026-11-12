# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Tutorial Series - IGRINS2 Data Reduction with DRAGONS'
copyright = '2027, Association of University for Research in Astronomy'
author = 'DRAGONS Team'

version = '5.0_dev'
#release = '5.0'
#rtdurl = 'v'+release
#release = '5.0.x'
#rtdurl = 'release-'+release
rtfurl = 'latest'

today = 'January 2027'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.imgmath'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = '.rst'
master_doc = 'index'

pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = 'TutorialSeries-IGRINS2DRDRAGONS'


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
.. |descriptors| replace:: :raw-html:`<a href="https://astrodata-user-manual.readthedocs.io/en/{v}/appendices/appendix_descriptors.html" target="_blank">descriptors</a>`
.. |descriptor| replace:: :raw-html:`<a href="https://astrodata-user-manual.readthedocs.io/en/{v}/appendices/appendix_descriptors.html" target="_blank">descriptors</a>`
.. |reduce| replace:: :raw-html:`<a href="https://dragons-recipe-system-users-manual.readthedocs.io/en/{v}/reduce.html" target="_blank">reduce</a>`
.. |showd| replace:: :raw-html:`<a href="https://dragons-recipe-system-users-manual.readthedocs.io/en/{v}/supptools.html#showd" target"_blank">showd</a>`
.. |showrecipes| replace:: :raw-html:`<a href="https://dragons-recipe-system-users-manual.readthedocs.io/en/{v}/supptools.html#showrecipes" target="_blank">showrecipes</a>`
.. |showpars| replace:: :raw-html:`<a href="https://dragons-recipe-system-users-manual.readthedocs.io/en/{v}/supptools.html#showpars" target="_blank">showpars</a>`
.. |typewalk| replace:: :raw-html:`<a href="https://dragons-recipe-system-users-manual.readthedocs.io/en/{v}/supptools.html#typewalk" target="_blank">typewalk</a>`
.. |atfile| replace:: :raw-html:`<a href="https://dragons-recipe-system-users-manual.readthedocs.io/en/{v}/reduce.html#the-file-facility" target="_blank">"at-file" Facility</a>`
.. |astrodatauser| replace:: :raw-html:`<a href="https://astrodata-user-manual.readthedocs.io/en/{v}/" target="_blank">Astrodata User Manual</a>`

.. |RSUserInstall| replace:: :raw-html:`<a href="https://dragons.readthedocs.io/projects/recipe-system-users-manual/en/{v}/install.html" target="_blank">DRAGONS Installation Instructions</a>`
.. |DRAGONS| replace:: :raw-html:`<a href="http://dragons.readthedocs.io/en/{v}/">DRAGONS</a>`


""".format(v = rtfurl)
