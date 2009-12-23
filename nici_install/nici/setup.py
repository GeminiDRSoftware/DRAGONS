#!/usr/bin/env python

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('nici',parent_package,top_path)

    # add gspline module
    #config.add_extension('gspline', ['gspline/gspline.pyf','gspline/gspline.c'])

    config.add_data_files('*.cl')
    config.add_data_files('*.par')
    config.add_data_files('bad_r.fits','bad_b.fits')
    #config.add_data_files('ncqlook')     # Will copy the links to a file
    #config.add_data_files('ncprepare')
    #config.add_data_files('ncmkflats')
    #config.add_data_files('ncscience')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration = configuration)
