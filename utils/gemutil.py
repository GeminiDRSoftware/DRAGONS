
from pyraf import iraf
from iraf import gemini
from iraf import gemlocal

import mefutil
#reload(mefutil)
import strutil
import time




"""This file contains the following utilities:
    imageName (image, rawpath='', prefix='auto', observatory='gemini-north',
        verbose=True)
    appendSuffix (filename, suffix)
    replaceSuffix (filename, suffix)
    gemdate (zone='UT')
    gemhedit (filename=None, extension=0, keyword="", value="", comment="")
    gimverify (image, sci_ext='SCI', dq_ext='DQ')
    printlog (text, logfile=None, verbose=True)
    removeExtension (images)
    appendFits (images)    
    chomp(line)
"""

    
# This is used by removeExtension(), appendSuffix(), and replaceSuffix(),
# which are defined in this file.
extensions = [".fits", ".fit", ".pl", ".imh", ".hhh", ".tab", ".dat"]

gimverify_does_not_exist = 0    # does not exist
gimverify_MEF = 1               # exists and is a MEF file
gimverify_simple_FITS = 2       # exists and is a simple FITS file
gimverify_pl = 3                # exists and is pixel list image
gimverify_imh = 4               # exists and is an imh image
gimverify_hhh = 5               # exists and is an hhh image
gimverify_other = 6             # exists but is not one of the above types
#---------------------------------------------------------------------------
def imageName (image, rawpath='', prefix='auto', observatory='gemini-north',
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
    
    #
    #  Make sure rawpath and mdfdir have a trailing "/"
    #
    if ((rawpath != "") and (rawpath[-1] != "/") and (rawpath[-1] != "$")):
        rawpath = rawpath + "/"
        #self.rawpath = rawpath
        #self.mdfdir = mdfdir
    #---------------------------------------------------------
    # Build file prefix for future use

    if prefix == "auto":
        observatory = observatory.lower()
        if observatory == "gemini-north":
            siteprefix = "N"
        elif observatory == "gemini-south":
            siteprefix = "S"
        else:
            print "# ERROR:   unknown observatory:", observatory
            bye()
        iraf.getfakeUT()
        utdate = iraf.getfakeUT.fakeUT
        prefix = siteprefix + utdate + "S" 


    #----------------------------------------------------------
    # Construct image name

    iraf.gemisnumber (image, "integer", verbose=iraf.no)
    if iraf.gemisnumber.fl_istype:
        if verbose: print "# IMAGENAME - Constructing image name based on today's UT date..."
        imstring = "%04d"%(int(image))
        image = prefix + imstring + ".fits"

    imagenorawpath = image

    # is there a directory name
    iraf.fparse(image)
    if iraf.fparse.directory == "":
        iraf.gimverify(image)
        if iraf.gimverify.status != 0:
            iraf.gimverify(rawpath+image)
        if iraf.gimverify.status == 1:
            print "# ERROR: Cannot access image", image
            #raise SystemExit
        elif iraf.gimverify.status != 0:
            print "# ERROR: Image %s is not a MEF file" % (l_image,)
            #raise SystemExit

    image = iraf.gimverify.outname
    fitsimage = image + ".fits"
        
    observer = mefutil.getkey ("observer", fitsimage)
    if verbose: print "# IMAGENAME - Observer ", observer

    if verbose: print "# IMAGENAME - Using image", image

    return fitsimage, imagenorawpath        

#---------------------------------------------------------------------------
def appendSuffix (filename, suffix):
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

    @param suffix: the suffix (e.g. "_flt") to append
    @type suffix: string

    @return: the input file name with the suffix included
    @rtype: string
    
    """

    found = False
    # extensions is a list of recognized filename extensions.
    for extn in extensions:
        if filename.endswith (extn):
            k = filename.rfind (extn)
            newname = filename[:k] + suffix + extn
            found = True
            break

    if not found:
        newname = filename + suffix

    return newname

#---------------------------------------------------------------------------
def replaceSuffix (filename, suffix):
    """Replace the suffix in the file name.

    If filename includes an underscore ("_") character, the slice
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

    found = False
    # extensions is a list of recognized filename extensions.
    for extn in extensions:
        if filename.endswith (extn):
            j = filename.rfind ("_")
            if j >= 0:
                newname = filename[:j] + suffix + extn
            else:
                k = filename.rfind (extn)
                newname = filename[:k] + suffix + extn
            found = True
            break

    if not found:
        j = filename.rfind ("_")
        if j >= 0:
            newname = filename[:j] + suffix
        else:
            newname = filename + suffix

    return newname

#---------------------------------------------------------------------------
def gemdate (zone="UT", timestamp = None):
    
    """Get the current date and time.

    @param zone: "UT" or "local", to indicate whether the time should be
        UTC (always standard time) or local time (which can be either
        standard or daylight saving time)
    @type zone: string

    @return: date and time formatted as "yyyy-mm-ddThh:mm:ss";
        every value is an integer
    @rtype: string    
    """

    if timestamp == None:
        if zone == "UT":        
            t = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        elif zone == "local":
            t = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        else:
            raise ValueError, "Invalid time zone = %s" % zone
    else:
        t = timestamp.strftime("%Y-%m-%dT%H:%M:%S")
    return t

#---------------------------------------------------------------------------
def gemhedit (filename=None, extension=0, keyword="", value="", comment="",
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

    fd = pyfits.open (filename, mode="update")
    header = fd[extension].header

    if delete:
        if header.has_key (keyword.upper()):
            del (header[keyword])
        # else do nothing
    else:
        if header.has_key (keyword.upper()):
            header[keyword] = value
        else:
            header.update (keyword, value=value, comment=comment)

    fd.close()

#---------------------------------------------------------------------------
def gimverify (image, sci_ext="SCI", dq_ext="DQ"):
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
    words = image.split ('[')
    if len (words) > 1:
        image = words[0]

    has_dq = False      # this is an initial value than can be reset below
    if not os.access (image, os.R_OK):
        return (gimverify_does_not_exist, has_dq)

    words = image.split ('.')
    if len (words) > 1:
        extension = words[-1]
    else:
        extension = ""

    if extension == "pl":
        type = gimverify_pl
    elif extension == "imh":
        type = gimverify_imh
    elif extension == "hhh":
        type = gimverify_hhh
    elif extension == "fits" or extension == "fit":
        # Find out what type of FITS file this is.
        fd = pyfits.open (image)
        if len (fd) > 1:
            type = gimverify_other      # may be reset below
            try:
                if fd[sci_ext].header["xtension"] == "IMAGE":
                    type = gimverify_MEF
                    try:
                        if fd[dq_ext].header["xtension"] == "IMAGE":
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
def printlog (text, logfile=None, verbose=True):
    """Append text to the log file.

    @param text: text string to log
    @type text: string

    @param logfile: name of log file, or None
    @type logfile: string

    @param verbose: if True, then also print to standard output
    @type verbose: boolean
    """

    if logfile == "STDOUT":
        logfile = None
        verbose = True

    if text[0:5] == "ERROR" or text[0:7] == "WARNING":
        verbose = True

    if logfile is not None:
        fd = open (logfile, mode="a")
        fd.write (text + "\n")
        fd.close()

    if verbose:
        print text

#---------------------------------------------------------------------------
def removeExtension (images):
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

    if isinstance (images, str):
        is_a_list = False
        images = [images]
    else:
        is_a_list = True
    modified = []
    for image in images:
        found = False
        # extensions is a list of recognized filename extensions.
        for extn in extensions:
            if image.endswith (extn):
                k = image.rfind (extn)
                modified.append (image[:k])
                found = True
                break
        if not found:
            modified.append (image)

    if is_a_list:
        return modified
    else:
        return modified[0]

#---------------------------------------------------------------------------
def appendFits (images):
    """
    !!!NOTE!!! This function calls the appendFits in strutil. Thus, if you want to use appendFits,
    use the one in there. This remains for backwards compatibility.
    
    Append ".fits" to each name in 'images' that lacks an extension.

    >>> print appendFits ('abc')
    abc.fits
    >>> print appendFits ('abc.fits')
    abc.fits
    >>> print appendFits (['abc', 'xyz.fits'])
    ['abc.fits', 'xyz.fits']

    @param images: a file name or a list of file names
    @type images: a string or a list of strings

    @return: the input file names with ".fits" appended to each, unless
        the name already ended in a recognized extension.
    @rtype: list of strings
    """

    return strutil.appendFits( images )
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
    
    return strutil.chomp( line )
    
