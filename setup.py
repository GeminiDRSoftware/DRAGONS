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
import re

from setuptools import setup, find_packages, Extension

from astrodata._version import version

try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

PACKAGENAME = 'dragons'

PACKAGES = find_packages('.', exclude=['*tests'])
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

# files for spec stds in geminidr/gemini/lookups/spectrophotometric_standards
for inst in instruments:
    for root, dirs, files in os.walk(os.path.join('geminidr', inst,
                                                  'lookups', 'spectrophotometric_standards')):
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

# Some .dat files are found in the lookup directories
for inst in instruments:
    for root, dirs, files in os.walk(os.path.join('geminidr', inst,
                                                  'lookups')):
        files = [f for f in files if not (f.endswith('.pyc') or f.endswith('.py'))]
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
    "gempy.library.cython_utils",
    [os.path.join('gempy', 'library', 'cython_utils.' + suffix)],
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
      ],
      install_requires = [
          'astropy',
          'astroquery',
          'future',
          'ginga',
          'imexam',
          'matplotlib',
          'numpy',
          'python-dateutil',
          'pytest',
          'scipy',
          'specutils',
          'sqlalchemy',
      ]
      )
