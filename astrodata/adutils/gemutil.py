import os
import sys

import mefutil
import strutil
import time

from astrodata.adutils import gemLog
from astrodata import Errors

log = None

"""This file contains the following utilities:
    imageName (image, rawpath='', prefix='auto', observatory='gemini-north',
        verbose=True)
    appendSuffix (filename, suffix)
    replaceSuffix (filename, suffix)
    gemdate (zone='UT')
    gemhedit (filename=None, extension=0, keyword='', value='', comment='')
    gimverify (image, sci_ext='SCI', dq_ext='DQ')
    printlog (text, logfile=None, verbose=True)
    removeExtension (images)
    appendFits (images)    
    chomp(line)
"""

def pyrafLoader():
    """
    This function is to load the modules needed by primitives that use pyraf. 
    It will also ensure there are no additional prints to the console when 
    loading the Gemini pyraf package.
    The loaded modules are to be added to the name-space of the primitive this
    function is called from, e.g., (pyraf, gemini, yes, no) = pyrafLoader()
    """
    import pyraf
    from pyraf import iraf
    from iraf import gemini
    # Required for stackFrames (gemcombine)
    from iraf import gemtools
    # Required (gireduce, giflat, gmosaic, gdisplay, gifringe)
    from iraf import gmos
    import StringIO
    
    # Changing the standard output so the excess prints while loading IRAF
    # packages does not get displayed
    SAVEOUT = sys.stdout
    capture = StringIO.StringIO()
    sys.stdout = capture
    
    # Setting the IRAF versions of True and False
    yes = iraf.yes
    no = iraf.no
    
    # Returning stdout back to normal so prints show on the screen
    sys.stdout = SAVEOUT
    
    return (pyraf, gemini, iraf.yes, iraf.no)

    
# This is used by removeExtension(), appendSuffix(), and replaceSuffix(),
# which are defined in this file.
extensions = ['.fits', '.fit', '.pl', '.imh', '.hhh', '.tab', '.dat']

gimverify_does_not_exist = 0    # does not exist
gimverify_MEF = 1               # exists and is a MEF file
gimverify_simple_FITS = 2       # exists and is a simple FITS file
gimverify_pl = 3                # exists and is pixel list image
gimverify_imh = 4               # exists and is an imh image
gimverify_hhh = 5               # exists and is an hhh image
gimverify_other = 6             # exists but is not one of the above types
#---------------------------------------------------------------------------
def imageName(image, rawpath='', prefix='auto', observatory='gemini-north',
               verbose=True):
    """Convert an image number to a filename using UT date. If image is already a string simply
       check image's existence and MEFness.

    @param image: image number if taken that night or image string
    @type image: string or int

    @param rawpath: location of image
    @type rawpath: string

    @param prefix: if 'auto' then prefix will be formatted as 'N20080801S'
    @type prefix: string

    @param observatory: name of observatory, can be either 'gemini-north' or 'gemini-south'
    @type observatory: string

    @param verbose: warnings will be printed
    @type verbose: Boolean

    @return: image name string with .fits
    @rtype: string
    """
    # retrieve global logger object
    global log
    if log==None:
        # instantiate the logger object and put into the global variable 
        log = gemLog.getGeminiLog()
        
    # loading and bringing the pyraf related modules into the name-space
    pyraf, gemini, yes, no = pyrafLoader()
    iraf = pyraf.iraf
    #
    #  Make sure rawpath and mdfdir have a trailing '/'
    #
    if ((rawpath != '') and (rawpath[-1] != '/') and (rawpath[-1] != '$')):
        rawpath = rawpath + '/'
    
    # Build file prefix for future use
    if prefix == 'auto':
        observatory = observatory.lower()
        if observatory == 'gemini-north':
            siteprefix = 'N'
        elif observatory == 'gemini-south':
            siteprefix = 'S'
        else:
            log.error('Observatory name passed in not gemini-north or '+
                      'gemini-south, it was '+ observatory, category='IQ')
            bye()
        # Creating a UT version of todays date following YYYYMMDD 
        iraf.getfakeUT()
        utdate = iraf.getfakeUT.fakeUT
        prefix = siteprefix + utdate + "S" 

    # Construct image name following one of three options
    
    # If input variable image is an integer
    # output image name will follow prefix(from above)+image+'.fits'
    iraf.gemisnumber(image, "integer", verbose=iraf.no)
    if iraf.gemisnumber.fl_istype:
        if verbose: print "# IMAGENAME - Constructing image name based on today's UT date..."
        imstring = "%04d"%(int(image))
        image = prefix + imstring + ".fits"

    imagenorawpath = image

    # Is there a directory name
    iraf.fparse(image)
    if iraf.fparse.directory == "":
        iraf.gimverify(image)
        if iraf.gimverify.status != 0:
            iraf.gimverify(rawpath+image)
        if iraf.gimverify.status == 1:
            log.error("Cannot access image "+image, category='IQ')
            #raise SystemExit
        elif iraf.gimverify.status != 0:
            log.error("Image %s is not a MEF file"% (l_image), category='IQ')
            #raise SystemExit

    image = iraf.gimverify.outname
    fitsimage = image + ".fits"
       
    observer = mefutil.getkey('observer', fitsimage)
    if verbose: 
        log.fullinfo('Observer of image '+imagenorawpath+
                             ' was found to be '+observer, category='IQ')

    return fitsimage, imagenorawpath        

#---------------------------------------------------------------------------
def appendSuffix(filename, suffix):
    """Append a suffix to the root of the file name.

    If filename does not end with one of the recognized extensions,
    the suffix will just be appended to filename.

    >>> print appendSuffix ('abc.fits', '_flt')
    abc_flt.fits
    >>> print appendSuffix ('abc', '_flt')
    abc_flt
    >>> print appendSuffix ('abc.xyz', '_flt')
    abc.xyz_flt

    @param filename: a file name
    @type filename: string

    @param suffix: the suffix (e.g. '_flt') to append
    @type suffix: string

    @return: the input file name with the suffix included
    @rtype: string
    
    """
    # retrieve global logger object
    global log
    if log==None:
        # instantiate the logger object and put into the global variable 
        log = gemLog.getGeminiLog()

    found = False
    # Extensions is a list of recognized filename extensions.
    for extn in extensions:
        if filename.endswith(extn):
            k = filename.rfind(extn)
            newname = filename[:k] + suffix + extn
            found = True
            break

    if not found:
        newname = filename + suffix
    log.status('appendSuffix changed the filename from '+filename+' to '+
               newname, category='IQ')
    return newname

#---------------------------------------------------------------------------
def replaceSuffix(filename, suffix):
    """Replace the suffix in the file name.

    If filename includes an underscore ('_') character, the slice
    between that point (the rightmost underscore) and the extension will
    be replaced with the specified suffix.  If there is no underscore,
    the suffix will be inserted before the extension.
    If filename does not end with one of the recognized extensions,
    all of the string starting with the rightmost underscore will be
    replaced by the specified suffix, or the suffix will be appended
    if there is no underscore in filename.

    >>> print replaceSuffix ('abc_raw.fits', '_flt')
    abc_flt.fits
    >>> print replaceSuffix ('abc.fits
    ', '_flt')
    abc_flt.fits
    >>> print replaceSuffix ('abc_raw.flub', '_flt')
    abc_flt
    >>> print replaceSuffix ('abc.flub', '_flt')
    abc.flub_flt

    @param filename: a file name
    @type filename: string

    @param suffix: the suffix to replace the existing suffix
    @type suffix: string

    @return: the input file name with the suffix included
    @rtype: string
    """
    # retrieve global logger object
    global log
    if log==None:
        # instantiate the logger object and put into the global variable 
        log = gemLog.getGeminiLog()
    
    found = False
    # Extensions is a list of recognized filename extensions.
    for extn in extensions:
        if filename.endswith(extn):
            j = filename.rfind('_')
            if j >= 0:
                newname = filename[:j] + suffix + extn
            else:
                k = filename.rfind(extn)
                newname = filename[:k] + suffix + extn
            found = True
            break

    if not found:
        j = filename.rfind('_')
        if j >= 0:
            newname = filename[:j] + suffix
        else:
            newname = filename + suffix
    
    log.status('replaceSuffix changed the filename from '+filename+' to '+
               newname, category='IQ')
    return newname

#---------------------------------------------------------------------------
def gemdate(zone='UT', timestamp = None):
    
    """Get the current date and time.

    @param zone: 'UT' or 'local', to indicate whether the time should be
        UTC (always standard time) or local time (which can be either
        standard or daylight saving time)
    @type zone: string

    @return: date and time formatted as 'yyyy-mm-ddThh:mm:ss';
        every value is an integer
    @rtype: string    
    """

    if timestamp == None:
        if zone == 'UT':        
            t = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())
        elif zone == 'local':
            t = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())
        else:
            raise ValueError, 'Invalid time zone = %s' % zone
    else:
        t = timestamp.strftime('%Y-%m-%dT%H:%M:%S')
    return t
    
def stdDateString(date):
    retstr = date.strftime('%Y-%m-%dT%H:%M:%S')
    return retstr

#---------------------------------------------------------------------------
def gemhedit(filename=None, extension=0, keyword='', value='', comment='',
              delete=False):
    """Modify or delete an existing keyword or add a new keyword to a header.

    @param filename: name of FITS file containing the header to be updated
    @type filename: string

    @param extension: extension number, EXTNAME string,
        or (EXTNAME, EXTVER) tuple
    @type extension: int, string, or tuple

    @param keyword: keyword name
    @type keyword: string

    @param value: value to assign to the keyword
    @type value: string, int, float, or boolean

    @param comment: comment for the keyword; this is only used if the
        keyword is not already present in the header
    @type comment: string

    @param delete: if True and the keyword is present in the header,
        delete the keyword
    @type delete: boolean
    """

    fd = pyfits.open(filename, mode='update')
    header = fd[extension].header

    if delete:
        if header.has_key (keyword.upper()):
            del (header[keyword])
        # else do nothing
    else:
        if header.has_key (keyword.upper()):
            header[keyword] = value
        else:
            header.update(keyword, value=value, comment=comment)

    fd.close()

#---------------------------------------------------------------------------
def gimverify(image, sci_ext='SCI', dq_ext='DQ'):
    """Check whether the specified image exists, and if so, get the type.

    @param image: name of an image; the name must not use wildcard characters
    @type image: string

    @param sci_ext: name or number of the extension for science image data
    @type sci_ext: string or int

    @param dq_ext: name or number of the extension for data quality flags
    @type dq_ext: string or int

    @return: (type, has_dq)
        type is an integer code indicating existence and image type:
            gimverify_does_not_exist - does not exist or no read access
            gimverify_MEF - exists and is a multi-extension FITS (MEF) file;
                for this case we check that the sci extension is an IMAGE
            gimverify_simple_FITS - exists and is a simple FITS file; for this
                case we check that there actually is a data block in the
                primary header/data unit
            gimverify_pl - exists and is an iraf pixel list image
            gimverify_imh - exists and is an imh image
            gimverify_hhh - exists and is an hhh (GEIS) image
            gimverify_other - exists but is not one of the above image types
        has_dq is a boolean flag (always False unless gimverify_MEF):
            true indicates that the file contains a DQ extension (IMAGE)
    @rtype: tuple of int and boolean
    """

    # If the image name includes an extension specification (for FITS or
    # hhh format), strip it off.  Note that if the name includes a wildcard
    # using brackets, this will fail because part of the file name will be
    # chopped off.
    
    # image at this point would be something like 'blablabla.fits[blabla]'
    words = image.split('[')
    if len(words) > 1:
        # Selecting the far left (zeroth) string found by the split function 
        # as the image name
        image = words[0]

    has_dq = False      # this is an initial value than can be reset below
    if not os.access(image, os.R_OK):
        # Note: all gimverify codes are initialized at the top of this module
        return (gimverify_does_not_exist, has_dq)

    # image at this point would be something like 'blablabla.fits'
    words = image.split('.')
    if len (words) > 1:
        # Selecting the far right string found by the split function as the 
        # file extension
        extension = words[-1]
    else:
        extension = ''
        
    # Setting the type to be returned based on what the file extensions is
    if extension == 'pl':
        type = gimverify_pl
    elif extension == 'imh':
        type = gimverify_imh
    elif extension == 'hhh':
        type = gimverify_hhh
    elif extension == 'fits' or extension == 'fit':
        # Find out what type of FITS file this is.
        fd = pyfits.open (image)
        if len(fd) > 1:
            type = gimverify_other      # may be reset below
            try:
                if fd[sci_ext].header['xtension'] == 'IMAGE':
                    type = gimverify_MEF
                    try:
                        if fd[dq_ext].header['xtension'] == 'IMAGE':
                            has_dq = True
                    except:
                        has_dq = False
            except KeyError:
                type = gimverify_other
        elif fd[0].data is not None:
            type = gimverify_simple_FITS
        else:
            type = gimverify_other
        fd.close()
    else:
        type = gimverify_other

    return (type, has_dq)

#---------------------------------------------------------------------------
def printlog(text, logfile=None, verbose=True):
    """Append text to the log file.

    @param text: text string to log
    @type text: string

    @param logfile: name of log file, or None
    @type logfile: string

    @param verbose: if True, then also print to standard output
    @type verbose: boolean
    """
    # retrieve global logger object
    global log
    if log==None:
        # instantiate the logger object and put into the global variable 
        log = gemLog.getGeminiLog()
        

    if logfile == 'STDOUT':
        logfile = None
        verbose = True

    if (text[0:5] == 'ERROR') or (text[0:7] == 'WARNING'):
        verbose = True

    if logfile is not None:
        fd = open(logfile, mode='a')
        fd.write(text + '\n')
        fd.close()

    if verbose:
        log.fullinfo(text, category='IQ')

#---------------------------------------------------------------------------
def removeExtension(images):
    """Remove the extension from each file name in the list.

    If a file name does not end with one of the recognized extensions
    (identified by a variable 'extensions' that is global to this file),
    the original file name will be included unchanged in the output list.

    >>> print removeExtension ('abc.fits')
    abc
    >>> print removeExtension (['abc.fits', 'def.imh', 'ghi.fit'])
    ['abc', 'def', 'ghi']

    @param images: a file name or a list of file names
    @type images: a string or a list of strings

    @return: the input file names with extensions removed
    @rtype: list of strings
    """

    if isinstance(images, str):
        is_a_list = False
        images = [images]
    else:
        is_a_list = True
    modified = []
    for image in images:
        found = False
        # extensions is a list of recognized filename extensions.
        for extn in extensions:
            if image.endswith(extn):
                k = image.rfind(extn)
                modified.append(image[:k])
                found = True
                break
        if not found:
            modified.append(image)

    if is_a_list:
        return modified
    else:
        return modified[0]

#---------------------------------------------------------------------------
def appendFits(images):
    """
    !!!NOTE!!! This function calls the appendFits in strutil. Thus, if you want to use appendFits,
    use the one in there. This remains for backwards compatibility.
    
    Append '.fits' to each name in 'images' that lacks an extension.

    >>> print appendFits ('abc')
    abc.fits
    >>> print appendFits ('abc.fits')
    abc.fits
    >>> print appendFits (['abc', 'xyz.fits'])
    ['abc.fits', 'xyz.fits']

    @param images: a file name or a list of file names
    @type images: a string or a list of strings

    @return: the input file names with '.fits' appended to each, unless
        the name already ended in a recognized extension.
    @rtype: list of strings
    """

    return strutil.appendFits(images)
#---------------------------------------------------------------------------

def chomp(line):
    """
    !!!NOTE!!! This function calls the chomp in strutil. Thus, if you want to use chomp,
    use the one in there. This remains for backwards compatibility.
    
    Removes newline(s) from end of line if present.
    
    @param line: A possible corrupted line of code
    @type line: str
    
    @return: Line without any '\n' at the end.
    @rtype: str
    """
    
    return strutil.chomp(line)
#---------------------------------------------------------------------------
   
   
def rename_hdu(name=None, ver=None, hdu=None):
    """
    :param name: New "EXTNAME" for the given extension.
    :type name: string
    
    :param ver: New "EXTVER" for the given extension
    :type ver: int

    The rename_ext() function is used in order to rename an HDU with a new
    EXTNAME and EXTVER based identifier.  Merely changing the EXTNAME and 
    EXTEVER values in the extensions pyfits.Header are not sufficient.
    Though the values change in the pyfits.Header object, there are special
    HDU class members which are not updated. 
    
    :warning:   This function maniplates private (or somewhat private)  HDU
                members, specifically "name" and "_extver". STSCI has been
                informed of the issue and
                has made a special HDU function for performing the renaming. 
                When generally available, this new function will be used instead of
                manipulating the  HDU's properties directly, and this function will 
                call the new pyfits.HDUList(..) function.
    """
    # @@TODO: change to use STSCI provided function.
    if hdu is None:
        raise Errors.gemutilError("ERROR: HDU required to rename hdu")
    if type(name) == tuple:
        ver = name[1]
        name = name[0]
    if ver == None:
        ver = 1
    nheader = hdu.header
    kafter = "GCOUNT"
    if nheader.get("TFIELDS"): 
        kafter = "TFIELDS"
    if name is None:
        if not nheader.has_key("EXTNAME"):
            raise Errors.gemutilError("name is None and EXTNAME not in header")
        nheader.update("extver", ver, "added by AstroData", after="EXTNAME")
    else:
        nheader.update("extname", name, "added by AstroData", after=kafter)
        nheader.update("extver", ver, "added by AstroData", after="EXTNAME")
    hdu.name = name
    hdu._extver = ver

