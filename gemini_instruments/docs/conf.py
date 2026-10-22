# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'gemini_instruments'
copyright = '2026, Association of Universities for Research in Astronomy'
author = 'DRAGONS Team'

version = '5.0_dev'
release = '5.0.0'
#rtdurl = 'v'+release
#release = '5.0.x'
#rtdurl = 'release-'+release
rtdurl = 'latest'

today = 'April 2026'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix of source filenames.
source_suffix = '.rst'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_logo = "images/DRAGONS - Icon blue.png"


# -- Customization -------------------------------------------------------------

# Activate the todos
todo_include_todos = True

def setup(app):
   app.add_css_file('custom_code.css')
   app.add_css_file('todo-styles.css')
   app.add_css_file('rtd_theme_overrides.css')
   app.add_css_file('fonts.css')


rst_epilog = """
.. role:: raw-html(raw)
   :format: html

.. |ADMaster| replace:: :raw-html:`<a href="http://astrodata.readthedocs.io/en/{v}">Astrodata Manual</a>`
.. |DRAGONS| replace:: :raw-html:`<a href="https://dragons.readthedocs.io/en/{v}/">DRAGONS</a>`

""".format(v = rtdurl)
