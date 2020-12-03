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

from setuptools import setup, find_packages, Extension

from astrodata._version import version

try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

PACKAGENAME = 'dragons'
PACKAGES = find_packages('.', exclude=['*tests*'])

# PACKAGE_DATA
PACKAGE_DATA = {
    'geminidr': ['geminidr/*/lookups/source_detection/*',
                 'geminidr/*/lookups/spectrophotometric_standards/*',
                 'geminidr/*/lookups/BPM/*',
                 'geminidr/*/lookups/MDF/*'],
    'gempy': ['gempy/numdisplay/*',
              'gempy/library/config/README'],
    'recipe_system': ['recipe_system/adcc/client/*'],
}

# SCRIPTS
SCRIPTS = [
    os.path.join('recipe_system', 'scripts', name)
    for name in ('adcc', 'caldb', 'reduce', 'superclean')
]
SCRIPTS += [
    os.path.join('gempy', 'scripts', name)
    for name in ('dataselect', 'fwhm_histogram', 'gmosn_fix_headers',
                 'gmoss_fix_HAM_BPMs.py', 'gmoss_fix_headers.py',
                 'pipeline2iraf', 'profile_all_obj', 'psf_plot', 'showrecipes',
                 'showd', 'showpars', 'swapper', 'typewalk', 'zp_histogram')
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
      version=version(),
      description='Gemini Data Processing Python Package',
      author='Gemini Data Processing Software Group',
      author_email='sus_inquiries@gemini.edu',
      url='http://www.gemini.edu',
      maintainer='Science User Support Department',
      license='BSD',
      zip_safe=False,
      packages=PACKAGES,
      package_data=PACKAGE_DATA,
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
          'astropy>=4.1',
          'astroquery',
          'future',
          'ginga',
          'gwcs>=0.14',
          'imexam',
          'matplotlib',
          'numpy',
          'python-dateutil',
          'scipy',
          'specutils',
          'sqlalchemy',
      ],
      extras_require={
          'docs': ['sphinx', 'sphinx_rtd_theme'],
          'test': ['pytest', 'pytest-remotedata', 'coverage', 'objgraph'],
      },
      project_urls={
          'Issue Tracker': 'https://github.com/GeminiDRSoftware/DRAGONS',
          'Documentation': 'https://dragons.readthedocs.io/',
      },
      # keywords=['astronomy', 'astrophysics', 'science', 'gemini'],
      python_requires='>=3.6',
      )
