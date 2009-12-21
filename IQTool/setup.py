#!/usr/bin/env python

"""
Setup script for IQTool for DAs.

In this module:

Usage:
  python setup.py install --prefix=/astro/iraf/i686/gempylocal
  python setup.py sdist
"""

import os.path
from distutils.core import setup

MODULENAME = 'IQTool'

# PACKAGES and PACKAGE_DIR
SUBMODULES = ['iq', 'utils']
PACKAGES = [MODULENAME]
for m in SUBMODULES:
    PACKAGES.append('.'.join([MODULENAME,m]))
PACKAGE_DIRS = {}
PACKAGE_DIRS['IQTool'] = '.'
PACKAGE_DIRS['utils'] = 'iq'


#PACKAGE_DATA
PACKAGE_DATA = {}
for p in PACKAGES:
    PACKAGE_DATA[p]=['Changes']

DATA_FILES = None

# SCRIPTS
PYGEM_SCRIPTS = [ os.path.join('lib','gemiq') ]
SCRIPTS = []
SCRIPTS.extend(PYGEM_SCRIPTS)

EXTENSIONS = None

setup ( name='IQTool',
        version='1.1',
        description='Image Quality tools',
        author='Gemini Observatory',
        author_email='jholt@gemini.edu',
        url='http://www.gemini.edu',
        maintainer='Kathleen Labrie',
        maintainer_email='klabrie@gemini.edu',
        packages=PACKAGES,
        package_dir=PACKAGE_DIRS,
        package_data=PACKAGE_DATA,
        data_files=DATA_FILES,
        scripts=SCRIPTS,
        ext_modules=EXTENSIONS,
        classifiers=[
            'Development Status :: Beta',
            'Intended Audience :: Gemini Instrument Scientists',
            'Operating System :: Linux :: RHEL',
            'Programming Language :: Python',
            'Topic :: Instrument Support',
            'Topic :: Instrument Checkouts',
            'Topic :: Engineering',
        ],
      )

