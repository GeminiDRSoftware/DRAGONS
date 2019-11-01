#!/usr/bin/env python

"""
Setup script for gemini_python

In this package:
    astrodata
    gemini_instruments
    geminidr
    gempy
    recipe_system

Usage:
    python setup.py install --prefix=/astro/iraf/URlocal-v1.5.2/pkgs/gemini_python-v2.0.0
    python setup.py
"""

import os
import os.path
import re

from setuptools import setup
from setuptools.extension import Extension

from astrodata import version

try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

PACKAGENAME = 'dragons'

# PACKAGES and PACKAGE_DIRS
ASTRODATA_MODULES = ['astrodata']

GEMINI_INST_MODULES = ['gemini_instruments',
                       'gemini_instruments.bhros',
                       'gemini_instruments.cirpass',
                       'gemini_instruments.f2',
                       'gemini_instruments.flamingos',
                       'gemini_instruments.gemini',
                       'gemini_instruments.gmos',
                       'gemini_instruments.gnirs',
                       'gemini_instruments.gpi',
                       'gemini_instruments.graces',
                       'gemini_instruments.gsaoi',
                       'gemini_instruments.hokupaa_quirc',
                       'gemini_instruments.hrwfs',
                       'gemini_instruments.igrins',
                       'gemini_instruments.michelle',
                       'gemini_instruments.nici',
                       'gemini_instruments.nifs',
                       'gemini_instruments.niri',
                       'gemini_instruments.oscir',
                       'gemini_instruments.phoenix',
                       'gemini_instruments.skycam',
                       'gemini_instruments.texes',
                       'gemini_instruments.trecs',
                       ]

GEMINIDR_MODULES = ['geminidr',
                    'geminidr.core',
                    'geminidr.f2',
                    'geminidr.f2.lookups',
                    'geminidr.f2.recipes',
                    'geminidr.f2.recipes.qa',
                    'geminidr.f2.recipes.sq',
                    'geminidr.gemini',
                    'geminidr.gemini.lookups',
                    'geminidr.gemini.lookups.source_detection',
                    'geminidr.gmos',
                    'geminidr.gmos.lookups',
                    'geminidr.gmos.recipes',
                    'geminidr.gmos.recipes.qa',
                    'geminidr.gmos.recipes.sq',
                    'geminidr.gnirs',
                    'geminidr.gnirs.lookups',
                    'geminidr.gnirs.recipes',
                    'geminidr.gnirs.recipes.qa',
                    'geminidr.gnirs.recipes.sq',
                    'geminidr.gsaoi',
                    'geminidr.gsaoi.lookups',
                    'geminidr.gsaoi.recipes',
                    'geminidr.gsaoi.recipes.qa',
                    'geminidr.gsaoi.recipes.sq',
                    'geminidr.niri',
                    'geminidr.niri.lookups',
                    'geminidr.niri.recipes',
                    'geminidr.niri.recipes.qa',
                    'geminidr.niri.recipes.sq',
                    ]

GEMPY_MODULES = ['gempy',
                 'gempy.adlibrary',
                 'gempy.eti_core',
                 'gempy.gemini',
                 'gempy.gemini.eti',
                 'gempy.library',
                 'gempy.library.config',
                 'gempy.utils',
                 ]

RS_MODULES = ['recipe_system',
              'recipe_system.adcc',
              'recipe_system.adcc.servers',
              'recipe_system.cal_service',
              'recipe_system.mappers',
              'recipe_system.reduction',
              'recipe_system.utils',
              ]

SUBMODULES = []
SUBMODULES.extend(ASTRODATA_MODULES)
SUBMODULES.extend(GEMINI_INST_MODULES)
SUBMODULES.extend(GEMINIDR_MODULES)
SUBMODULES.extend(GEMPY_MODULES)
SUBMODULES.extend(RS_MODULES)

PACKAGES = []
PACKAGES.extend(SUBMODULES)
PACKAGE_DIRS = {}
PACKAGE_DIRS[''] = '.'

# PACKAGE_DATA
PACKAGE_DATA = {}

# sextractor files in geminidr/gemini/lookups/source_detection
gemdrdir = re.compile('geminidr/')
PACKAGE_DATA['geminidr'] = []
instruments = ['gemini', 'f2', 'gmos', 'gnirs', 'gsaoi', 'niri']
for inst in instruments:
    for root, dirs, files in os.walk(os.path.join('geminidr', inst,
                                                  'lookups', 'source_detection')):
        files = [f for f in files if not f.endswith('.pyc')]
        # remove the 'geminidr/' part of the file paths, then add to the DATA
        PACKAGE_DATA['geminidr'].extend(
            map((lambda f: os.path.join(gemdrdir.sub('', root), f)), files)
        )

# BPMs and MDFs throughout the geminidr package
for inst in instruments:
    for root, dirs, files in os.walk(os.path.join('geminidr', inst,
                                                  'lookups', 'BPM')):
        files = [f for f in files if not f.endswith('.pyc')]
        if len(files) > 0:
            # remove the 'geminidr/' part, add to DATA
            PACKAGE_DATA['geminidr'].extend(
                map((lambda f: os.path.join(gemdrdir.sub('', root), f)), files)
            )
    for root, dirs, files in os.walk(os.path.join('geminidr', inst,
                                                  'lookups', 'MDF')):
        files = [f for f in files if not f.endswith('.pyc')]
        if len(files) > 0:
            # remove the 'geminidr/' part, add to DATA
            PACKAGE_DATA['geminidr'].extend(
                map((lambda f: os.path.join(gemdrdir.sub('', root), f)), files)
            )

# GUI
rsdir = re.compile('recipe_system/')
PACKAGE_DATA['recipe_system'] = []
for root, dirs, files in os.walk(os.path.join('recipe_system', 'adcc',
                                              'client')):
    # remove the 'recipe_system/' part, add to DATA
    PACKAGE_DATA['recipe_system'].extend(
        map((lambda f: os.path.join(rsdir.sub('', root), f)), files)
    )

# Numdisplay support files
PACKAGE_DATA['gempy'] = []
for f in ('ichar.dat', 'imtoolrc', 'README'):
    PACKAGE_DATA['gempy'].append(os.path.join('numdisplay', f))
PACKAGE_DATA['gempy'].append(os.path.join('library', 'config', 'README'))

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# For packaging, need to add tests and docs
#    (also see the 'sdist' section in the
#     old setup.py.  handles symlinks)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
DATA_FILES = []

# SCRIPTS
RS_SCRIPTS = [os.path.join('recipe_system', 'scripts', 'adcc'),
              os.path.join('recipe_system', 'scripts', 'caldb'),
              os.path.join('recipe_system', 'scripts', 'reduce'),
              os.path.join('recipe_system', 'scripts', 'superclean'),
              ]

GEMPY_SCRIPTS = [
    os.path.join('gempy', 'scripts', 'dataselect'),
    os.path.join('gempy', 'scripts', 'fwhm_histogram'),
    os.path.join('gempy', 'scripts', 'gmosn_fix_headers'),
    os.path.join('gempy', 'scripts', 'gmoss_fix_HAM_BPMs.py'),
    os.path.join('gempy', 'scripts', 'gmoss_fix_headers.py'),
    os.path.join('gempy', 'scripts', 'pipeline2iraf'),
    os.path.join('gempy', 'scripts', 'profile_all_obj'),
    os.path.join('gempy', 'scripts', 'psf_plot'),
    os.path.join('gempy', 'scripts', 'showrecipes'),
    os.path.join('gempy', 'scripts', 'showd'),
    os.path.join('gempy', 'scripts', 'showpars'),
    os.path.join('gempy', 'scripts', 'swapper'),
    os.path.join('gempy', 'scripts', 'typewalk'),
    os.path.join('gempy', 'scripts', 'zp_histogram'),
]
SCRIPTS = []
SCRIPTS.extend(RS_SCRIPTS)
SCRIPTS.extend(GEMPY_SCRIPTS)

EXTENSIONS = []

if use_cython:
    suffix = 'pyx'
else:
    suffix = 'c'
cyextensions = [Extension(
    "gempy.library.cyclip",
    [os.path.join('gempy', 'library', 'cyclip.' + suffix)],
),
]
if use_cython:
    CYTHON_EXTENSIONS = cythonize(cyextensions)
else:
    CYTHON_EXTENSIONS = cyextensions

EXTENSIONS.extend(CYTHON_EXTENSIONS)

setup(name='dragons',
      version=version(),
      description='Gemini Data Processing Python Package',
      author='Gemini Data Processing Software Group',
      author_email='sus_inquiries@gemini.edu',
      url='http://www.gemini.edu',
      maintainer='Science User Support Department',
      packages=PACKAGES,
      package_dir=PACKAGE_DIRS,
      package_data=PACKAGE_DATA,
      data_files=DATA_FILES,
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
      ]
      )
