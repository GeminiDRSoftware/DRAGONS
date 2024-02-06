#!/usr/bin/env python

"""
Setup script for gemini_python

In this package:
    astrodata
    gemini_instruments
    geminidr
    gempy
    recipe_system

Usage: pip install [-e] .

"""

import os

from setuptools import setup, find_packages, Extension

try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

VERSION = '3.1.0-dev'

PACKAGENAME = 'dragons'
PACKAGES = find_packages()

# SCRIPTS
SCRIPTS = [
    os.path.join('recipe_system', 'scripts', name)
    for name in ('adcc', 'caldb', 'reduce', 'superclean', 'provenance')
]
SCRIPTS += [
    os.path.join('gempy', 'scripts', name)
    for name in ('dataselect', 'dgsplot', 'fwhm_histogram', 'gmosn_fix_headers',
                 'gmoss_fix_HAM_BPMs.py', 'gmoss_fix_headers.py',
                 'pipeline2iraf', 'profile_all_obj', 'psf_plot', 'showrecipes',
                 'showd', 'showpars', 'typewalk', 'zp_histogram')
]

# EXTENSIONS
suffix = 'pyx' if use_cython else 'c'
EXTENSIONS = [
    Extension("gempy.library.cython_utils",
              [os.path.join('gempy', 'library', 'cython_utils.' + suffix)])
]
if use_cython:
    EXTENSIONS = cythonize(EXTENSIONS)

setup(name='dragons',
      version=VERSION,
      description='Gemini Data Processing Python Package',
      author='Gemini Data Processing Software Group',
      author_email='sus_inquiries@gemini.edu',
      url='http://www.gemini.edu',
      maintainer='Science User Support Department',
      license='BSD',
      zip_safe=False,
      packages=PACKAGES,
      include_package_data=True,
      scripts=SCRIPTS,
      ext_modules=EXTENSIONS,
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Gemini Ops',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: POSIX :: Linux',
          'Operating System :: Linux :: CentOS',
          'Operating System :: MacOS :: MacOS X',
          'Programming Language :: Python',
          'Topic :: Gemini',
          'Topic :: Data Reduction',
          'Topic :: Scientific/Engineering :: Astronomy',
      ],
      install_requires=[
          'asdf',
          'astropy>=4.3',
          'astroquery',
          'astroscrappy',
          'bokeh>=3',
          'bottleneck',
          'future',
          'gwcs>=0.14',
          'holoviews',
          'matplotlib',
          'numpy',
          'psutil',
          'python-dateutil',
          'requests',
          'scipy',
          'specutils>=1.1',
          'sqlalchemy',
      ],
      extras_require={
          'all': ['ginga', 'imexam'],
          'docs': ['sphinx', 'sphinx_rtd_theme'],
          'test': ['pytest', 'pytest_dragons>=1.0.0', 'coverage', 'objgraph'],
      },
      project_urls={
          'Issue Tracker': 'https://github.com/GeminiDRSoftware/DRAGONS',
          'Documentation': 'https://dragons.readthedocs.io/',
      },
      # keywords=['astronomy', 'astrophysics', 'science', 'gemini'],
      python_requires='>=3.7',
      )
