#!/usr/bin/env python

"""
Setup script for gemplotlib.
"""

#import os.path

from distutils.core import setup

PACKAGES = ['gemplotlib']

PACKAGE_DIRS = {}
PACKAGE_DIRS['gemplotlib'] = "."

DATA_FILES = None

# SCRIPTS
GEMPLOTLIB_SCRIPTS = [ 'overlay.py' ]
SCRIPTS = GEMPLOTLIB_SCRIPTS

EXTENSIONS = None

setup ( name='gemplotlib',
        version='0.9',
        description='Gemini Python Plotting Tools',
        author='Gemini Observatory',
        author_email='jholt@gemini.edu',
        url='http://www.gemini.edu',
        maintainer='Jen Holt',
        maintainer_email='jholt@gemini.edu',
        packages=PACKAGES,
        package_dir=PACKAGE_DIRS,
        data_files=DATA_FILES,
        scripts=SCRIPTS,
        ext_modules=EXTENSIONS,
        classifiers=[
            'Development Status :: Beta',
            'Intended Audience :: Gemini IQ Working Group',
            'Operating System :: Linux :: RHEL',
            'Programming Language :: Python',
            'Topic :: Instrument Support',
            'Topic :: Engineering',
       ],
     )
