#!/usr/bin/env python
"""
Script to fix the headers of GMOS-S Hamamatsu CCD data.

Fixes the following keywprds:

    PHU -
        DATE-OBS
        RELEASE
        OBSEPOCH

    Image Extensions -
        CCDSEC
        CRPIX1

This script can be executed as a command line script. Or the function
'correct_headers' can be imported and a HDUList supplied to use it as part of a
Python script.

Contents:

    Functions
    =========
        main
        parse_command_line_inputs
        correct_headers

        fix_phu
        fix_date_obs
        fix_release_date
        fix_obsepoch
        
        fix_image_extensions
        fix_ccdsec
        fix_crpix1
        
        _update_date_object
        _get_key_value
        _parse_section
        _get_time
        _default_log_file
        _parse_files
        _expand_inputs
        _parse_file_system
        _parse_file_system_inputs

TODO:
    Add more logging / reports
    
"""
import argparse
import datetime
import glob
import logging
import os
import re
import shutil
import sys

try:
    import pyfits as pf
except ImportError:
    try:
        from astropy.io import fits as pf
    except ImportError:
        raise ImportError("Cannot import pyfits or astropy.io.fits, "
                          "please make sure one of them is installed.")

from astropy.time import Time

####
# Script meta-data
__version__ = [0, 4, 0]
__PROGRAM__ = os.path.splitext(os.path.basename(__file__))[0]
__VERSION_STRING__ = '.'.join([str(x) for x in __version__])

# Define visible functions and constants to user when if they import *
all = ["main", "parse_command_line_inputs", "correct_headers", "__version__"]

####
# Regular expressions
DATE_STRING = "(?P<year>[0-9]{4})-(?P<month>[01][0-9])-(?P<day>[0123][0-9])"
SECTION_STRING = ("\[(?P<X1>[0-9]+)\:(?P<X2>[0-9]+),"
                  "(?P<Y1>[0-9]+):(?P<Y2>[0-9]+)\]")
FILE_SYSTEM_PATHS = ["rawpath", "outpath", "backup_path"]
FILE_SYSTEM_REGEXP = re.compile("[\$\~]+")

# String constants for asthetics
DELIMINATOR = "===="

# Global used for logging
DATETIME = None

# Observatory program and observation type look ups
__OBSCLASS_NON_PROGRAM_CALS__ = ["daycal", "acqcal", "partnercal"]
__OBSCLASS_PROGRAM_OBS__ = ["science", "acq", "progcal"]
__SCIENCE_PROGRAM_IDS__ = ["Q", "DD", "C", "LP", "FT", "SV"]
__SV_PROGRAM_IDS__ = ["SV"]

# Used to update PHU date related keywords
__CORRECT_KEY__ = "DATE"
__CORRECT_TIME_KEY__ = "UT"

####
# Instrument specfic constants
INSTRUMENT = "GMOS-S"
DETTYPE = "S10892"
CCDWIDTH = 2048
CCDGAP = 61
CENTRE_X_OFFSET = 0.5
CENTRE_X_DETECTOR = 2048 + 1024
ADDITIONAL_OFFSET = 0  # Apparent offset not in CD_MATRIX - As of the WCS
                       # upgrade during the week of 2014-06-29 --> 2014-07-05
                       # this is now 0. Prior to that the CD matrix is wrong
                       # addtional 36 pixels required to get it close to be
                       # correct in the outer two CCDs.

####
# The wrapper function called by main(); it is this function should be imported
# if required to be part of another Python script
def correct_headers(hdulist, report=None, logger=None, correct_phu=True,
                    correct_image_extensions=True):
    """
    The hard work function to update the headers. Updates the PHU and then
    image extensions, in place! The keywords updated are in the doc string for
    this module. In addition most of the functions are named with the keyword
    that the function is to modify in it.

    This is only for early GMOS-S Hamamatsu data. There are criteria to meet to
    fix the header:

        Keywords in PHU:
            INSTRUME == 'GMOS-S'
            DETTYPE == 'S10892'

    `hdulist`: PyFITS HDUList with the first element must be a PrimaryHDU

    `logger`: Logging object. Default will create it's own logging object.

    Returns:
        None if crtieria for updating an image are not met but keywords exist

        or
        
        bool: whether the file was updated or not (Currently all files
        that match the criteria will AWLAYS be updated).

    Raises:
        KeyError if criteria keywords are None
        
    """ 
    assert isinstance(hdulist, pf.HDUList)

    global log
    
    # Create logger
    if logger is None:
        log = get_logger("/dev/null/", "INFO")
    else:
        log = logger

    # Perform checks
    phu = hdulist[0]
    instrument = _get_key_value(phu, 'INSTRUME')
    dettype = _get_key_value(phu, 'DETTYPE')

    log.debug ("{0}".format(hdulist.filename()))
    log.debug ("INSTRUMENT: {0}; DETTYPE: {1}".format(instrument, dettype))

    # Check for non-existent keywords
    if None in [instrument, dettype]:
        errmsg = ("Keyord has not been found - INSTRUMENT: {0}; "
                  "DETTYPE: {1}".format(instrument, dettype))
        log.error(errmsg)
        raise KeyError(errmsg)

    # Check values of criteria
    if (instrument != INSTRUMENT or dettype != DETTYPE):
        log.info("Criteria not met")
        return None

    # Call function to fix PHU keywords and image extension keywords
    phu_updated = False
    if correct_phu:
        phu_updated = fix_phu(hdulist[0])
        
    image_extensions_updated = False
    if correct_image_extensions:
        image_extensions_updated = fix_image_extensions(hdulist)
        
    if True in [phu_updated, image_extensions_updated]:
        updated = True
    else:
        updated = False

    return updated

####
# Image pixel extension fixes
def fix_image_extensions(hdulist):
    """ Fix image extension keywords """
    assert isinstance(hdulist, pf.HDUList)

    updated = False
    for i, hdu in enumerate(hdulist[1:]):
        if isinstance(hdu, pf.ImageHDU):
            if True in [fix_ccdsec(hdu), fix_crpix1(hdu)]:
                updated = True

    return updated


def fix_ccdsec(hdu):
    """ Fix CCDSEC keywords in image extensions """
    section_regexp = re.compile(SECTION_STRING)
    
    # In unbinned space
    ccdsec = _get_key_value(hdu, 'CCDSEC')
    detsec = _get_key_value(hdu, 'DETSEC')

    if None in [ccdsec, detsec]:
        raise ValueError("CCDSEC {0}; detsec {1}".format(ccdsec, detsec))

    updated = False
    ccd_coords = list(section_regexp.match(ccdsec).groups())
    detector_coords = list(section_regexp.match(detsec).groups())

    # Y coordinates should match!
    if ccd_coords[2:4] != detector_coords[2:4]:
        raise ValueError("Y values: {0} {1}".format(ccdsec, detsec))
    
    # X coordinates maybe wrong
    if ccd_coords[0:2] != detector_coords[0:2]:

        for i, x in enumerate(detector_coords[0:2]):
            offset_x = int(x) - CCDWIDTH
            if offset_x <= 0:
                if ccd_coords[i] != detector_coords[i]:
                    # Use DETSEC
                    ccd_coords[i] = detector_coords[i]
                    updated = True
                else:
                    # Reset offset to x
                    offset_x = x
            elif offset_x > CCDWIDTH:
                updated = True
                offset_x -= CCDWIDTH
            # update ccd_coords
            ccd_coords[i] = offset_x
        # Reset CCDSEC
        ccdsec = "[{0}:{1},{2}:{3}]".format(ccd_coords[0],
                                            ccd_coords[1],
                                            ccd_coords[2],
                                            ccd_coords[3])
        hdu.header['CCDSEC'] = ccdsec

    return updated


def fix_crpix1(hdu):
    """ Fix crpix1 keywords in image extensions """
    # Always updated
    updated = True
    crpix1 = _get_key_value(hdu, 'CRPIX1')
    
    datasec = _get_key_value(hdu, 'DATASEC')
    detsec = _get_key_value(hdu, 'DETSEC')
    biassec = _get_key_value(hdu, 'BIASSEC')
    ccdsum = _get_key_value(hdu, 'CCDSUM')

    log.debug("datasec: {0}".format(datasec))
    log.debug("detsec: {0}".format(detsec))
    log.debug("biassec: {0}".format(biassec))
    log.debug("ccdsum: {0}".format(ccdsum))

    data_coords = _parse_section(datasec)
    detector_coords = _parse_section(detsec)
    biassec_coords = _parse_section(biassec)
    [xbin, ybin] = [int(value) for value in ccdsum.replace(" ", "")]

    if biassec_coords[0] == 1:
        offset = biassec_coords[1] - (biassec_coords[0] - 1)
    else:
        offset = 0

    # IRAF Coords
    new_crpix1 = (((CENTRE_X_DETECTOR - (detector_coords[0] - 1)) /
                   xbin) + CENTRE_X_OFFSET)

    new_crpix1 += offset
    
    if detector_coords[0] < CCDWIDTH:
        factor = 1
    elif detector_coords[0] > 2 * CCDWIDTH:
        factor = -1
    else:
        factor = 0

    if factor != 0 and ADDITIONAL_OFFSET is not None:
        additional_offset = ADDITIONAL_OFFSET
    else:
        additional_offset = 0

    new_crpix1 += (factor * (CCDGAP + additional_offset)) / xbin

    hdu.header['CRPIX1'] = new_crpix1

    return updated


####
# PHU fixes
def fix_phu(phu):
    """ Fix the incorrect keywords in the PHU """
    assert isinstance(phu, pf.PrimaryHDU)

    updated = False
    if True in [fix_date_obs(phu),
                fix_release_date(phu),
                fix_obsepoch(phu)]:
        updated = True
    return updated


def fix_date_obs(phu, correct_key=__CORRECT_KEY__):
    """ Correct DATE-OBS keyword in PHU"""
    date = _get_key_value(phu, "DATE")
    date_obs = _get_key_value(phu, "DATE-OBS")

    if None in [date, date_obs]:
        raise ValueError("DATE: {0}; DATE-OBS{1}".format(date, date_obs))
    regexp = re.compile(DATE_STRING)
    date_match = regexp.match(date)
    date_obs_match = regexp.match(date_obs)

    updated = False
    # Check their formats are correct
    if None in [date_match, date_obs_match]:
        # Return PHU untouched
        raise ValueError("_DATE: {0}; DATE-OBS: {1}".format(date, date_obs))
    elif date != date_obs:
        update_value = False
        for i, match in enumerate(date_match.groups()):
            # Check that the day and month match
            if match != date_obs_match.groups()[i]:
                if i == 0:
                    update_value = True
                else:
                    # Do something here for user to record it
                    pass
        if update_value:
            phu.header["DATE-OBS"] = date
            updated = True

    return updated


def fix_release_date(phu, correct_key=__CORRECT_KEY__):
    """
    Fix release date keyword in PHU
    
    Crteria:
    
    If PROGID is CAL or OBSCLASS is {dayCal,acqCal,partnerCal} -> 'Release Now'

    If PROGID is -Q-,-DD-,-C- and OBSCLASS {science,acq,progcal} -> '18 months'

    if PROGID is -SV- and obsclass {science,acq,progcal} -> '3 months'

    """
    date = _get_key_value(phu, correct_key)
    gemini_program_id = _get_key_value(phu, "GEMPRGID")
    obsclass = _get_key_value(phu, "OBSCLASS")

    release_date = datetime.datetime.strptime(date, "%Y-%m-%d")

    delta_time = 18
    if (obsclass.lower() in __OBSCLASS_NON_PROGRAM_CALS__ or
        "cal" in gemini_program_id.lower()):
        # Immediate release
        delta_time = 0
    elif obsclass.lower() in __OBSCLASS_PROGRAM_OBS__:
        # Proprietary time of some length
        if any(_id in gemini_program_id.upper()
               for _id in __SV_PROGRAM_IDS__):
            # 3 months
            delta_time = 3
        elif any(_id in gemini_program_id.upper()
                 for _id in __SCIENCE_PROGRAM_IDS__):
            # 18 months
            delta_time = 18

    if delta_time > 0:
        release_date = _update_date_object(release_date, delta_time)

    release_date = release_date.strftime("%Y-%m-%d")
    phu.header["RELEASE"] = release_date

    return True


def _update_date_object(date, months):
    """
    Employed here is the datetime 'timedelta' class.
    Because a 'month' is inherently fuzzy, timedelta does not use
    'months' to express a delta-t. Any timedelta arguments must be in 
    'days', 'weeks' (and others, see doc on datetime.timedelta).

    Conversion of months to days is fraught, but here a year == 365.25d
    and the 'months' parameter is converted to a rounded value from
    a calculated fraction(s) of a defined year.

    Eg., For 3 months:
         3/12 --> .25 * 365.25 --> round(91.3125) --> 91 days
         For 18 months:
         18/12 --> 1.5 * 365.25 --> round(547.857) --> 548 days

    This may introduce some minor variations over the course of a year, 
    and certainly for a leap year, but variations are not expected to 
    be > ~ (+/-)1d. These are perceived variations in that they arise
    from the common notion that one month is 30 days. 

    :param date: the date from which to add the timedelta for future release.
    :type date: <datetime.datetime>
    
    :param months: the number of months in the future.
    :type months: <int>

    :returns: The calculated future release date.
    :rtype: <datetime.datetime>
    
    """
    days_future = round(365.25 * (months / 12.0))
    delta_t = datetime.timedelta(days=days_future)
    return (date + delta_t)


def fix_obsepoch(phu, date_key=__CORRECT_KEY__, time_key=__CORRECT_TIME_KEY__):
    """
    OBSEPOCH is the EPOCH of the image. For this to be wrong the DATE-OBS
    must have been wrong. Which likely means the accuracy of the fix to this
    keyword will be limited to half a second rather than milli-seconds, as the
    DATE-OBS is normally supplied by the GPS time from the syncro bus. Both
    date and time are required to fix this keyword.
    
    """
    date = _get_key_value(phu, date_key)
    ut = _get_key_value(phu, time_key)
    ut_type = _get_key_value(phu, "TIMESYS")
    if None in [date, ut]:
        raise ValueError

    # Form an astropy time object
    date_time_string = ' '.join([date, ut])
    observation_utc = Time(date_time_string, scale=ut_type.lower())

    # Calculate the Julian EPOCH
    new_obsepoch = round(observation_utc.jyear, 11)

    try:
        orig_comment = phu.header.cards["OBSEPOCH"][2]
    except:
        orig_comment = "UPDATED"

    new_comment = ("{0}: from {1} and {2} "
                   "".format(orig_comment, date_key, time_key))

    phu.header["OBSEPOCH"] = (new_obsepoch, new_comment)

    return True
    
####
# Helper functions
def _get_key_value(header, key):
    """
    Helper function to get header keywords wrapping any KeyErrors
    
    Returns:
        Keyword value; None if KeyError

    """
    try:
        value = header.header[key]
    except KeyError:
        value = None
    return value
    

def _parse_section(section):
    section_regexp = re.compile(SECTION_STRING)
    return [int(value)
            for value in list(section_regexp.match(section).groups())]
        
def _get_time():
    global DATETIME
    if DATETIME is None:
        DATETIME = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    return DATETIME

def _default_log_file():
    """ Set the default log file name"""

    extension = "log"
    part_one = __PROGRAM__
    part_two = __VERSION_STRING__
    part_three = _get_time()             
#    components = [part_one, part_two, part_three]
    components = [part_one, part_three]
    logfile = '.'.join(["_".join(components), extension])
    
    return logfile

def _write_report(filename, report_name, message):
    """ Write filename and message to report_name on disk """ 
    report_file = open(report_name, 'w')
    report_file.write('\n'.join([filename, message]))
    report_file.close()
    

####
# Main function when called as script: Requires arg parsed input.
def main(args=None):
    """
    Handles the user inputs, creating back ups etc. before calling
    `correct_headers` where the HDUList is edited in place.

    Parameters:

    `args`: argparse.Namespace object

    Raises:
        IOError when trying to copy files

    """
    assert isinstance(args, argparse.Namespace)

    global log
    
    # Get the logger
    log = get_logger(args.logfile, args.log_level)

    backup_path = args.backup_path
    out_path = args.outpath

    # Print statements are for INFO to screen formatting only
    print ("")
    log.info("Logfile: {0}".format(args.logfile))
    print ("")
    log.info("Path: {0}".format(args.rawpath))
    log.info("Out path: {0}".format(out_path))
    log.info("Backup path: {0}".format(backup_path))

    # Create filenames and update headers
    for infile in args.files:
        do_update = False
        
        (inpath, filename) = os.path.split(infile)

        print ("")
        log.info("{0}".format(DELIMINATOR))
        print ("")
        log.info("File: {0}".format(filename))
        
        backup = os.path.join(backup_path, '.'.join([filename, args.suffix]))
        outfile = os.path.join(out_path, filename)

        log.debug("Infile: {0}".format(infile))
        log.debug("Backup: {0}".format(backup))
        log.debug("Outfile: {0}".format(outfile))

        # Check for existing files
        if not os.path.isfile(infile):
            log.warning('"{0}" does not exist'.format(infile))
        elif os.path.isfile(backup):
            log.warning('Backup file "{0}" already exists'.format(backup))
        elif infile != outfile and os.path.isfile(outfile):
            log.warning('"{0}" already exists'.format(outfile))
        else:
            do_update = True
        
        if not do_update:
            log.info("{0} has been left untouched".format(filename))
            continue

        # Report individual files if an error is raised
        report = '_'.join([args.prefix, os.path.splitext(filename)[0],
                           _get_time()])
        report = "{0}.{1}".format(report, "log")

        # Always open in readonly unless not doing a dry run and editing fie in
        # place 
        read_mode = 'readonly'
        
        # Make a backup of input file
        if not args.dry_run:
            try:
                log.info("Creating backup of {0}".format(filename))
                shutil.copy2(infile, backup)
            except IOError as err:
                errmsg = ("ERROR - type: {0}, type number: {1}, "
                          "message: {2}".format(type(err), err.errno,
                                                  err.strerror))
                _write_report(filename, report, errmsg)
                log.warning(errmsg)
                log.info("{0} has been left untouched".format(filename))
                del err
                continue

            if infile == outfile:
                read_mode = 'update'

        # Open the input file
        # Do not scale the data as we want it to be left untouched
        # This script deals with the backups
        try:
            hdulist = pf.open(infile, mode=read_mode, save_backup=False,
                              do_not_scale_image_data=True)
        except Exception as err:
            errmsg = ("ERROR - type: {0}, message: {1}".format(
                      type(err), str(err)))
            _write_report(filename, report, errmsg)
            log.warning(errmsg)
            log.info("{0} has been left untouched".format(filename))
            del err
            continue

        # Edits the HDUList in place
        try:
            updated = correct_headers(hdulist, report=report,
                                      logger=log, correct_phu=args.fix_phu,
                                      correct_image_extensions=args.fix_exts)
        except (IOError, ValueError, KeyError) as err:
            errmsg = ("ERROR - type: {0}, message: {1}".format(
                      type(err), str(err)))
            _write_report(filename, report, errmsg)
            log.warning(errmsg)
            del err
            continue
        
        if updated is not None and infile != outfile:
            if not args.dry_run:
                hdulist.writeto(outfile)
        elif updated is None:
            log.info("{0} has been left untouched".format(filename))

        # If the file has been opened in update mode:
        #     If updated the changes will be flushed on write if not it should
        #     be left untouched
        hdulist.close()

    print ("")
    log.info(DELIMINATOR)
    print ("")


####
# OS Directory Checking / Parsing
def _parse_files(args):
    """
    Reads the files and rawpath from args to determine the files to process.
    Specific checks on the file are performed by the 'correct_headers'
    function. Updates the argparse.Namesspace input inplace

    Parameters:
        args: argparse.Namespace

    Returns:
        argparse.Namespace
        
    """
    assert isinstance(args, argparse.Namespace)

    path = args.rawpath
    file_system_files = []
    for infile in args.files:
        found_files = glob.glob(os.path.join(path, infile.strip()))
        found_files = [fname for fname in found_files if fname != path]
        if found_files:                    
            file_system_files.extend(found_files)

    args.files = file_system_files
    
    return args


def _expand_inputs(variable):
    """
    Expand any environment or 'user' variables

    Parameters
        variable: basestring

    Returns:
        Expanded variable

    """
    if "$" in variable:
        variable = os.path.expandvars(variable)
        
    variable = os.path.expanduser(variable)
    return variable


def _parse_file_system(arg):
    """
    Determines if arg requires OS expansion or not.

    Parameters:
        `arg`: basestring

    Returns:
        Expanded input: basestring
    
    """
    if FILE_SYSTEM_REGEXP.search(arg):
        arg = _expand_inputs(arg)
    return arg


def _parse_file_system_inputs(args):
    """
    Parse paths given within argparse.Namespace object. Checks if variables
    require expanding. Finally parses file argument to determine list of
    request files. Updates the argpare.ArgumentParser input in place.

    Parameters:

        `args`: argparse.Namespace

    Returns:

        argpare.ArgumentParser

    Raises:
        IOError if a supplied path does not exist

    """
    # Parse the paths
    for test_path in FILE_SYSTEM_PATHS:
        path = _parse_file_system(args.__dict__[test_path])
        if os.path.isdir(path):
            args.__dict__[test_path] = _parse_file_system(
                    args.__dict__[test_path])
        else:
            raise IOError("Path \"{0}\" does not exist".format(path))

    # Parse the files
    args = _parse_files(args)

    return args


####
# Logging
log = None
LOGFORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
STDERRFMT = '%(message)s'

def get_logger(name=None, log_level="DEBUG"):
    """
    Get a custom logger

    Parameters:

        `name`: str, filename to log to

        `log_level`: Level of logging calls to print to screen

    Returns:

        logging.logger

    Raises:
    
        ValueError on incorrect logging level
    
    """
    # Get logger
    logger = logging.getLogger('')
    # Configure logger
    logger = _configure_log(logger, name, log_level)
    return logger


def _configure_log(logger, logfile=None, log_level="DEBUG"):
    """
    Configure a custom logger. If logfile is None a default logfile name is
    used.

    Parameters:

        `logger`: logging.logger

        `logfile`: string

        `log_level`: str; ['DEBUG', 'INFO']

    returns logging.logger

    """
    logging_level = getattr(logging, log_level.upper(), None)
    if not isinstance(logging_level, int):
        raise ValueError('Invalid log level: {0}'.format(logging_level))

    # Define handler for debug function handler
    def _arghandler(args=None, levelnum=None, prefix=None):
        largs = list(args)
        slargs = str(largs[0]).split('\n')
        for line in slargs:
            if prefix is not None:
                line = "{0}{1}".format(prefix, line)
            if len(line) == 0:
                log.log(levelnum, '')
            else:
                log.log(levelnum, line)

    # Define handler for DEBUG
    def _cdebug(*args):
        _arghandler(args, 10)

    # Create logger object with name of module
    log = logging.getLogger(__PROGRAM__)
    setattr(log, 'debug', _cdebug)

    # Log will always record debug statements; but only log_level will go to
    # screen
    logging.basicConfig(level="DEBUG", format=LOGFORMAT,
        datefmt='%Y-%m-%d %H:%M:%S', filename=logfile, filemode='w')

    # Define a Handler which writes logging_level messages or higher to the
    # sys.stderr
    screen = logging.StreamHandler()
    screen.setLevel(logging_level)

    # Set different format for STDERR
    formatter = logging.Formatter(STDERRFMT)
    
    # add the handler to the root logger
    log.addHandler(screen)

    return log

####
# Command line parsing
def parse_command_line_inputs():
    """
    Command line parser. This can be inputed and called with a string 'acting'
    like a command line.

    Raises:
        IOError; if supplied files / file regular expression when checked on
                 disk is empty.

    """
    parser = argparse.ArgumentParser(description="Correct GMOS-S Hamamatsu "
                                     "Data Headers", prog=__PROGRAM__)
    parser.add_argument('--files', '-f', dest="files", action="append",
                        default=[], help=("Files to update; unix style "
                                          "wildcards allowed - if quoted"),
                        required=True)

    parser.add_argument('--path', dest="rawpath", action="store", default="./",
                        help="Path to raw data")

    parser.add_argument('--destination', dest="outpath", action="store",
                        default="./", help="Path to put updated data in")

    parser.add_argument('--backup_path', dest="backup_path", action="store",
                        default="./", help="Path to store backups")

    parser.add_argument('--dry-run', '-n', dest="dry_run", action="store_true",
                        default=False, help=("Do not create output other "
                                             "than logfile"))

    parser.add_argument('--suffix', '-s', dest="suffix", action="store",
                        default="ORIG", help=("Suffix applied to copied "
                                              "input 'original' files"))
    parser.add_argument('--prefix', '-p', dest="prefix", action="store",
                        default=__PROGRAM__, help=("prefix for reports; "
                                                   "created in the pwd and "
                                                   " only if an exception is "
                                                   "raised"))

    parser.add_argument('--nofix_phu', dest="fix_phu", action="store_false",
                        default=True, help=("Do not fix PHU"))

    parser.add_argument('--nofix_exts', dest="fix_exts", action="store_false",
                        default=True, help=("Do not fix image extensions"))

    parser.add_argument('--logfile', dest="logfile", action="store",
                        default=_default_log_file(), help="Logfile")

    parser.add_argument('--log_level', dest="log_level", action="store",
                        default="INFO", help="Logging verbosity level")

    parser.add_argument('--version', action="version",
                        version="{0} {1}".format(__PROGRAM__,
                                                 __VERSION_STRING__))

    args = parser.parse_args()

    # Parse the file and path arguments
    args = _parse_file_system_inputs(args)

    if not args.files:
        raise IOError("No input files found or supplied")
    return args

####
# Run as a script    
if __name__ == "__main__":
    """TODO return correct status"""

    # Parse arguments: Allow exceptions to be raised as the parsing is done up
    # front
    try:
        args = parse_command_line_inputs()
    except KeyboardInterrupt:
        # Parsing arguments may take a long time allow user to escape easily
        print "KeyboardInterrupt"
        sys.exit(1)

    # Call the main function to update headers
    # Only catch KeyboardInterrupts - Specific error handling in 'main' /
    # 'correct_headers' to allow a long running process.
    try:
        main(args)
    except KeyboardInterrupt:
        print "KeyboardInterrupt"
        sys.exit(1)
