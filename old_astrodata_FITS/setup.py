#!/usr/bin/env python

"""
Setup script for astrodata_FITS

The Recipes, Primitives and astrodata configurations in this package are
developed by the Gemini Data Processing Software Group

In this package:
   ADCONFIG_FITS: Types, Descriptors and astrodata configurations for
                    FITS data.

Usage:
  python setup.py install --prefix=/astro/iraf/i686/gempylocal
  python setup.py sdist
"""

import os.path
import re
import glob

from distutils.core import setup

svndir = re.compile('.svn')

PACKAGENAME = 'astrodata_FITS'
VERSION = '0.1.0'
#RECIPENAME = 'RECIPES_FITS'
CONFIGNAME = 'ADCONFIG_FITS'
#PIFNAME = 'PIF_FITS'

# PACKAGES and PACKAGE_DIRS
PACKAGES = [PACKAGENAME]
# uncomment below if recipes and primitives are eventually added
#PACKAGES.append('.'.join([PACKAGENAME,RECIPENAME]))
#PACKAGES.append('.'.join([PACKAGENAME,RECIPENAME,'primitives']))
#PACKAGES.append('.'.join([PACKAGENAME,PIFNAME]))
#PACKAGES.append('.'.join([PACKAGENAME,PIFNAME,'piffits']))
#slash = re.compile('/')
#for root, dirs, files in os.walk(os.path.join(PIFNAME,'piffits')):
#    if not svndir.search(root) and len(files) > 0:
#        pifmodules = map((lambda d: slash.sub('.','/'.join([PACKAGENAME,root,d]))),\
#                         filter((lambda d: not svndir.search(d)), dirs))
#        PACKAGES.extend( pifmodules )

PACKAGE_DIRS = {}
PACKAGE_DIRS[PACKAGENAME] = '.'

# PACKAGE_DATA
PACKAGE_DATA = {}
PACKAGE_DATA[PACKAGENAME] = []
#for s in ['.']+[RECIPENAME]:
#    PACKAGE_DATA[PACKAGENAME].extend([os.path.join(s,'Copyright'),
#                                     os.path.join(s,'ReleaseNote'),
#                                     os.path.join(s,'README'),
#                                     os.path.join(s,'INSTALL'),
#                                     ])
#PACKAGE_DATA[PACKAGENAME].extend(glob.glob(os.path.join(RECIPENAME,'recipe.*')))
#PACKAGE_DATA[PACKAGENAME].extend(glob.glob(os.path.join(RECIPENAME,'subrecipes','recipe.*')))

# uncomment below as necessary.  For now, only 'descriptors' are found in this package
#PACKAGE_DATA[PACKAGENAME].append(os.path.join(CONFIGNAME,'structures','*.py'))
#for root, dirs, files in os.walk(os.path.join(CONFIGNAME,'lookups')):
#    if not svndir.search(root) and len(files) > 0:
#        PACKAGE_DATA[PACKAGENAME].extend( map((lambda f: os.path.join(root, f)), files) )
for root, dirs, files in os.walk(os.path.join(CONFIGNAME,'descriptors')):
    if not svndir.search(root) and len(files) > 0:
        PACKAGE_DATA[PACKAGENAME].extend( map((lambda f: os.path.join(root, f)), files) )
#for root, dirs, files in os.walk(os.path.join(CONFIGNAME,'classifications')):
#    if not svndir.search(root) and len(files) > 0:
#        PACKAGE_DATA[PACKAGENAME].extend( map((lambda f: os.path.join(root, f)), files) )

# DATA_DIRS and DATA_FILES
DATA_FILES = []

# DOC_DIR = os.path.join('share','astrodata_FITS')
# for root, dirs, files in os.walk(os.path.join('doc')):
#     if not svndir.search(root) and len(files) > 0:
#         dest = root.split('/',1)[1] if len(root.split('/',1)) > 1 else ""
#         DOC_FILES = map((lambda f: os.path.join(root,f)), files)
#         DATA_FILES.append( (os.path.join(DOC_DIR,dest), DOC_FILES) )


# SCRIPTS
SCRIPTS = []

EXTENSIONS = None

setup ( name=PACKAGENAME,
        version=VERSION,
        description='FITS Recipes, Primitives and AstroData configurations)',
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
            'Topic :: Astrodata'
            ],
       )
