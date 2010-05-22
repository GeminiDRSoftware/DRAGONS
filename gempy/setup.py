#!/usr/bin/env python

"""
Setup script for gempy.

The tools and modules in this package are developed by the Gemini Data Processing Software Group.

In this module:
  instruments:  Instrument specific tools or functions

Usage:
  python setup.py install --prefix=/astro/iraf/i686/gempylocal
  python setup.py sdist
"""

import os.path

from distutils.core import setup

MODULENAME = 'gempy'

# PACKAGES and PACKAGE_DIRS
SUBMODULES = ['instruments']
PACKAGES = [MODULENAME]
for m in SUBMODULES:
    PACKAGES.append('.'.join([MODULENAME,m]))
PACKAGE_DIRS = {}
PACKAGE_DIRS['gempy'] = '.'

# PACKAGE_DATA
PACKAGE_DATA = {}
for p in PACKAGES:
    PACKAGE_DATA[p]=['Copyright',
                     'ReleaseNote',
                     'README',
                     'INSTALL',
                     ]

DATA_FILES = None

# SCRIPTS
#GEMPY_SCRIPTS = [ os.path.join('iqtool','iqtool'),
#                 ]
SCRIPTS = []
#SCRIPTS.extend(GEMPY_SCRIPTS)

EXTENSIONS = None

setup ( name='gempy',
        version='0.1.0',
        description='Gemini Data Reduction Software',
        author='Gemini Data Processing Software Group',
        author_email='klabrie@gemini.edu',
        url='http://www.gemini.edu',
        maintainer='Gemini Data Processing Software Group',
        packages=PACKAGES,
        package_dir=PACKAGE_DIRS,
        package_data=PACKAGE_DATA,
        data_files=DATA_FILES,
        scripts=SCRIPTS,
        ext_modules=EXTENSIONS,
        classifiers=[
            'Development Status :: Beta',
            'Intended Audience :: Gemini Staff',
            'Operating System :: Linux :: RHEL',
            'Programming Language :: Python',
            'Topic :: Gemini',
            'Topic :: Data Reduction',
            'Topic :: Astronomy',
            ],
      )