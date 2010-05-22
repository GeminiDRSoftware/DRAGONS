#!/usr/bin/env python

"""
Setup script for astrodata_Gemini

The Recipes, Primitives and astrodata configurations in this package are
developed by the Gemini Data Processing Software Group

In this package:
   RECIPES_Gemini:  Recipes and Primitives for Gemini data
   ADCONFIG_Gemini: Types, Descriptors and astrodata configurations for
                    Gemini data.

Usage:
  python setup.py install --prefix=/astro/iraf/i686/gempylocal
  python setup.py sdist
"""

import os.path
import re
import glob

from distutils.core import setup

PACKAGENAME = 'astrodata_Gemini'
VERSION = '0.1.0'
MODULENAME = 'RECIPES_Gemini'

#PACKAGES and PACKAGE_DIRS
SUBMODULES = ['primitives']
PACKAGES = [MODULENAME]
for m in SUBMODULES:
    PACKAGES.append('.'.join([MODULENAME,m]))
PACKAGE_DIRS = {}
PACKAGE_DIRS[MODULENAME] = '.'


# PACKAGE_DATA
PACKAGE_DATA = {}
PACKAGE_DATA[MODULENAME] = []
for s in ['.']+SUBMODULES:
    PACKAGE_DATA[MODULENAME].extend([os.path.join(s,'Copyright'),
                                     os.path.join(s,'ReleaseNote'),
                                     os.path.join(s,'README'),
                                     os.path.join(s,'INSTALL'),
                                     ])
PACKAGE_DATA[MODULENAME].extend(glob.glob('recipe.*'))
PACKAGE_DATA[MODULENAME].append('primitives/primitives_List.txt')

print PACKAGE_DATA

# DATA_DIRS and DATA_FILES
DATA_FILES = []
DOC_DIR = os.path.join('doc','astrodata_Gemini','Recipes')
svndir = re.compile('.svn')
for root, dirs, files in os.walk('doc'):
    if not svndir.search(root) and len(files) > 0:
        dest = root.split('/',1)[1] if len(root.split('/',1)) > 1 else ""
        DOC_FILES = map((lambda f: os.path.join(root,f)), files)
        DATA_FILES.append( (os.path.join(DOC_DIR,dest), DOC_FILES) )

# SCRIPTS
SCRIPTS = []

EXTENSIONS = None

setup ( name=PACKAGENAME,
        version=VERSION,
        description='Gemini Recipes, Primitives and AstroData configurations)',
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
            'Development Status :: Development',
            'Intended Audience :: Gemini Staff',
            'Operating System :: Linux :: RHEL',
            'Programming Language :: Python',
            'Topic :: Gemini',
            'Topic :: Data Reduction',
            'Topic :: Astronomy',
            ],
       )
