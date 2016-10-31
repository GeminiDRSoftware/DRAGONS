#!/usr/bin/env python
"""
Contains functions to update the BPMs for GMOS-S Hamamatsu CCDs for the Gemini
Python package that have been copied from the Gemini IRAF package, which have
the bottom 48 rows (unbinned) trimmed.

Contents:
    main
    parse_command_line_inputs
    update_bpms

"""
import argparse
import os
import sys

import numpy as np
import pyfits as pf

from glob import glob

__DEFAULT_BPM_DTYPE__ = np.uint16
__PYFITS_UPDATE_MODES__ = ['update']
__TRIMMED_ROWS__ = 48
VERBOSE = False

__all__ = ["update_bpms"]

####
# Main functions
####
def update_bpms(ffiles):
    """
    The hard working driver function to update the BPMs for GMOS-S
    Hamamatsu CCDs for the Gemini Python package that have been copied from the
    Gemini IRAF package, which have the bottom 48 rows (unbinned) trimmed.

    Parameters
    ----------
    ffiles: List of populated data abstraction objects

    Returns
    -------
    ffiles: List of updated data abstraction objects

    """
    # Loop over files and call hidden function to perform the updates
    return [_update_bpm_file(infile) for infile in ffiles]


def _update_bpm_file(ffile):
    """
    Update an individual BPM

    Parameters
    ----------
    ffile: individual data abstraction object

    Returns
    -------
    ffile: Updated data abstarction object

    """
    # Loop over extensions
    # ASSUMES all extensions are image extensions AND first extension is a PHU
    
    # Unbinned versions are in entire detector and individual ccd frames
    unbinned_section_keywords = ['detsec', 'ccdsec']
    binned_section_keywords = ['datasec', 'trimsec']
    section_keywords = []
    section_keywords.extend(unbinned_section_keywords)
    section_keywords.extend(binned_section_keywords)

    print "Working on...\n{0}:\n".format(ffile.filename())
    
    ffile_slice = ffile[1:]
    for ext in ffile_slice:

        if VERBOSE:
            print "{0}, {1}\n".format(ext.name, ext.ver)
        
        # Require shape for some reason...
        old_shape = ext.data.shape
        old_y_size, old_x_size = old_shape
        
        # Parse sections and binning
        for key in section_keywords:
            vars()[key] = _parse_iraf_section(_get_key_value(ext, key.upper()))
        [x_bin, y_bin] = [int(value)
                          for value in _get_key_value(ext, 'CCDSUM').split()]

        # Updated array must cover entire raw amplifier
        # Set default array value to 1 == bad pixels
        # Add the bottom __TRIMMED_ROWS__ / y_bin
        old_array_start = __TRIMMED_ROWS__ / y_bin
        (new_y_size, new_x_size) = (old_y_size + old_array_start, old_x_size)
        new_size = (new_y_size, new_x_size)
        new_array = np.ones(new_size, dtype=__DEFAULT_BPM_DTYPE__)

        # Insert old data into new_array
        y_slice = slice(old_array_start, new_y_size)
        new_array[y_slice, :] = ext.data
        ext.data = new_array
        

        # Update keywords
        #
        # Binned versions
        for section in binned_section_keywords:
            value = vars()[section]
            old_str_value = _get_key_value(ext, section.upper())
            new_value = [0, new_y_size, value[2], value[3]]
            _set_key_value(ext, section.upper(),
                           _set_iraf_section(new_value))
            if VERBOSE:
                print ("{0}: {1} -> {2}\n".format(section.upper(),
                        old_str_value, _get_key_value(ext, section.upper())))

        # Unbinned version
        # ASSUMES that original y section end at physical end of CCD
        for section in unbinned_section_keywords:
            value = vars()[section]
            new_value = [0, value[1], value[2], value[3]]
            old_str_value = _get_key_value(ext, section.upper())
            _set_key_value(ext, section.upper(),
                           _set_iraf_section(new_value))
            if VERBOSE:
                print ("{0}: {1} -> {2}\n".format(section.upper(),
                        old_str_value, _get_key_value(ext, section.upper())))

    return ffile


####
# Hidden main functions
####
def _set_verbose(value):
    """ Set global VERBOSE variable """
    global VERBOSE
    VERBOSE = value


def _write_files(files, prefix=None, clobber=False):
    """ Write objects to disk """
    [_write_file(infile, prefix, clobber) for infile in files]


def _write_file(inobject, prefix=None, clobber=False):
    """
    Write the data abstraction object to a FITS file on disk

    Parameters
    ----------
    inobject: data abstraction class <PyFITS, astropy.io.FITS>

    prefix: String to be append to front of file name


    Returns
    -------

    None
    
    """
    local_write = 'writeto'
    local_write_func = getattr(inobject, local_write)
    filename = os.path.basename(inobject.filename())
    if prefix:
        filename = ''.join([prefix, filename])
    local_write_func(filename, clobber=clobber)
    return

    
def _open(args):
    """ Parse input arguments and then open """
    directory = args.directory
    if directory is None:
        directory = os.getcwd()

    files = []
    [files.extend(glob(os.path.join(directory, infile)))
                 for infile in args.infiles]
    return _open_files(files, args.open_mode)

    
def _open_files(inputs, mode):
    """
    Load the FITS files into the data abstraction class

    Parameters
    ----------
    inputs: FITS files to open <str>, <list>

    mode: mode to open file in; as used by the data abstraction close reader

    Returns
    -------

    open_list: list of data abstraction classes
    
    """
    assert isinstance(inputs, list)

    local_open = pf.open
    return [local_open(file, mode=mode) for file in inputs]

    
####
# Helper functions
####
# Header manipulation
def _get_key_value(ext, key, raise_exceptions=False):
    """
    Helper function to get header keywords wrapping any KeyErrors
    
    Returns:
        Keyword value; None if KeyError

    """
    err = None
    try:
        value = ext.header[key]
    except KeyError as err:
        value = None
    except Error as err:
        pass
    finally:
        if raise_exceptions and err is not None:
            raise err
    return value

    
def _set_key_value(ext, key, value):
    """
    Helper function to get header keywords wrapping any KeyErrors
    
    Returns:
        Keyword value; None if KeyError

    """
    ext.header[key] = value

# IRAF handling functions
def _set_iraf_section(section_list):
    """
    Convert python list containing section information into string using IRAF
    syntax

    """
    assert len(section_list) == 4
    [y1, y2, x1, x2] = section_list

    y1 += 1
    x1 += 1

    return "[{0:d}:{1:d},{2:d}:{3:d}]".format(x1, x2, y1, y2)

    
def _parse_iraf_section(section):
    """
    Convert IRAF section to Python syntax: zero based non-inclusive

    Returns
    -------

    [y1, y2, x1, x2]

    """
    assert isinstance(section, basestring)
        
    [x1, x2, y1, y2] = [int(a) for b in section.strip('[]').split(':')
                        for a in b.split(",")]
    x1 -= 1
    y1 -= 1
        
    return [y1, y2, x1, x2]

    
####
# Script handling
####
# Script functions
def main(args):
    """
    Open, update (write; determines whether to or not) and close BPM FITS
    files.

    Parameters
    ----------
    args: argpare parser object
    
    Returns
    -------
    None

    """
    write_files = args.no_write is False
    ffiles = _open(args)
    ffiles = update_bpms(ffiles)
    if write_files:
        _write_files(ffiles, args.prefix, args.clobber)

    return

def parse_command_line_inputs():
    parser = argparse.ArgumentParser(
        description="Update GMOS Hamamatsu BPMs from GIRAF for Gemini Python")
    parser.add_argument('--no_write', dest="no_write", action="store_true",
                        default=False,
                        help=("Do not write updated files to disk"))
    parser.add_argument('--mode', dest="open_mode", action="store",
                        default='readonly',
                        help=("Mode to open input files in"))
    parser.add_argument('--directory', dest="directory", action="store",
                        default=None,
                        help=("Location of input files"))
    parser.add_argument('--verbose', dest="verbose", action="store_true",
                        help=("verbosity"))
    parser.add_argument('--files', dest="infiles", nargs="+",
                        default=[], required=True,
                        help=("BPM FITS files to update"))
    group = parser.add_mutually_exclusive_group()

    group.add_argument('--clobber', dest="clobber", action="store_true",
                       default=False,
                       help=("Clobber existing files"))
    group.add_argument('--prefix', dest="prefix", action="store",
                       help=("Prefix to use when writing files to disk"))

    args = parser.parse_args()
    if not args.no_write and not args.clobber:
        assert isinstance(args.prefix, basestring), \
            ('When writing files and not clobbering, prefix must be set')

    # Set global verbose flag
    _set_verbose(args.verbose)
    return args

if __name__ == '__main__':
    """ Script parser """

    args = parse_command_line_inputs()
    main(args)
