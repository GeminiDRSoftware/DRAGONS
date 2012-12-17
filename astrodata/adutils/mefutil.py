
import os
import pyfits
from astrodata import new_pyfits_version

"""This file contains the following utilities:
    hdrhedit (header=None, keyword="", value="", comment="")
    putdata (filename, data, mef=None, phdr=None, hdr=None, clobber=False)
    getdata (filename)
    convertMEF (filename, template=None)
    getkeys (keywords, filename, extension=0, default="not found",
             must_exist=False)
    getkey (keyword, filename, extension=0, default="not found",
            must_exist=False)
    mefNumSciextheader = None)
    """
#---------------------------------------------------------------------------

def hdrhedit (header=None, keyword="", value="", comment=""):
    """Modify an existing keyword or add a new keyword to a header.

    @param header: update the keyword in this header
    @type header: pyfits Header object

    @param keyword: keyword name
    @type keyword: string

    @param value: value to assign to the keyword
    @type value: string, int, float, or boolean

    @param comment: comment for the keyword; this is only used if the
        keyword is not already present in the header
    @type comment: string
    """

    if keyword.upper() in header:
        header[keyword] = value
    else:
        if new_pyfits_version:
            header.update = header.set
        header.update (keyword, value=value, comment=comment)

#---------------------------------------------------------------------------

def putdata (filename, data, mef=None, phdr=None, hdr=None, clobber=False):
    """Write an image tonsciext = header['NSCIEXT'] a FITS file.

    The boolean mef argument lets you specify explicitly whether the
    file to be written should be simple FITS (primary HDU only) or
    multi-extension FITS (image in the first extension HDU).  If mef is
    None, this will be determined by whether the hdr argument is None.
    If hdr is None, phdr and data will be written to the primary HDU.
    If hdr is not None, only phdr will be written to the primary HDU, and
    hdr and data will be written to the first extension.fits

    @param filename: name of the FITS file
    @type filename: string

    @param data: image data
    @type data: array

    @param mef: if not None, explicitly specifies whether the output should
        be multi-extension FITS (mef=True) or simple FITS (mef=False)
    @type mef: boolean

    @param phdr: primary header
    @type phdr: pyfits Header object

    @param hdr: extension header or None
    @type hdr: pyfits Header object

    @param clobber: an existing file can be overwritten if clobber=True
    @type clobber: boolean
    """

    if mef is None:
        mef = (hdr is not None)

    if mef:
        # put the image in the first extension
        phdu = pyfits.PrimaryHDU (header=phdr)
        hdu = pyfits.ImageHDU (data, hdr)
        hdulist = pyfits.HDUList ([phdu, hdu])
    else:
        # put the image in the primary HDU
        phdu = pyfits.PrimaryHDU (data, phdr)
        hdulist = pyfits.HDUList (phdu)

    if clobber:
        os.remove (filename)

    hdulist.writeto (filename, output_verify="fix", clobber=clobber)
#---------------------------------------------------------------------------
def getdata (filename):
    """Read an image from a FITS file.

    The image will be read from the primary header/data unit if the
    primary HDU contains a data block; in this case the third element
    (the extension header) of the function value will be None.
    If the primary HDU consists of only a header, the image will be
    read from the first extension; in this case the extension header
    will also be returned (i.e. will not be None).

    @param filename: name of the FITS file
    @type filename: string

    @return: data, primary header, and either extension header or None
    @rtype: tuple with three elements
    """

    fd = pyfits.open (filename)
    phdr = fd[0].header
    if fd[0].data is not None:
        # take the image from the primary HDU
        hdr = None
        data = fd[0].data
    else:
        # take the image from the first extension HDU
        hdr = fd[1].header
        data = fd[1].data

    return (data, phdr, hdr)


#---------------------------------------------------------------------------

def convertMEF (filenames, output, extname=["SCI"], template=None):
    """Convert filenames to multi-extension FITS

    @param filenames: names of input FITS file
    @type filenames: string or list of strings

    @param output: name of output FITS file; if this file already exists,
        it will be overwritten
    @type output: string

    @param extname: extension names
    @type extname: string or list of strings

    @param template: name of input file whose primary header should be
        copied to the output primary header
    @type template: string
    """

    # take primary header for output from template
    fd = pyfits.open (template)
    phdu = pyfits.PrimaryHDU (header=fd[0].header)
    hdulist = pyfits.HDUList ([phdu])
    fd.close()

    if isinstance (filenames, str):
        filenames = [filenames]
    if isinstance (extname, str):
        extname = [extname]

    n_files = len (filenames)
    if len (extname) != n_files:
        print "input files and extension names =", filenames, extname
        raise RuntimeError, "lengths of lists must be the same"

    for n in range (n_files):
        fname = filenames[n]
        ename = extname[n]

        # hdr will be None, since 'fname' is a simple FITS file
        (data, phdr, hdr) = getdata (fname)

        # use primary header of 'fname' as extension header
        hdr = phdr.copy()

        # put the image in the extension
        if new_pyfits_version:
            hdr.update = hdr.set
        hdr.update ("EXTNAME", ename)
        hdr.update ("EXTVER", 1)
        hdu = pyfits.ImageHDU (data, hdr)
        hdulist.append (hdu)

    if os.access (output, os.F_OK):
        os.remove (output)
    hdulist.writeto (output, output_verify="fix")
#---------------------------------------------------------------------------

def getkeys (keywords, filename, extension=0, default="not found",
             must_exist=False):
    """Get keyword values from a FITS header.

    @param keywords: the names of the keywords to read from the header
    @type keywords: list of strings

    @param filename: name of the FITS file
    @type filename: string

    @param extension: extension number, EXTNAME string,
        or (EXTNAME, EXTVER) tuple
    @type extension: int, string, or tuple

    @param default: value to assign for the value of missing keywords
    @type default: string

    @param must_exist: True if it is an error for any keyword in the
        list to be missing
    @type must_exist: boolean

    @return: tuple of values, one for each keyword in the input keywords list
    @rtype: tuple
    """

    fd = pyfits.open (filename)
    hdr = fd[extension].header
    fd.close()

    results = []
    missing = []
    if new_pyfits_version:
        keywords_in_header = hdr.keys()
    else:
        cardlist = hdr.ascardlist()
        keywords_in_header = cardlist.keys()
    for keyword in keywords:
        keyword = keyword.upper()
        if keyword in keywords_in_header:
            value = hdr[keyword]
        else:
            missing.append (keyword)
            value = default
        results.append (value)

    if must_exist and missing:
        raise RuntimeError, "Missing keywords = %s" % repr (missing)

    return results
#---------------------------------------------------------------------------

def getkey (keyword, filename, extension=0, default="not found",
            must_exist=False):

    keyword = keyword.upper()
    results = getkeys ([keyword], filename=filename, extension=extension,
                       default=default, must_exist=must_exist)

    return results[0]
#---------------------------------------------------------------------------
def mefNumSciext(header=None):
    """Check to see if file is multi-extension fits (mef)
    @param header
    @type header: pyfits Header object
    """  
    num = header['NSCIEXT']    
    if  num < 2:
        raise 'CRITICAL:Header indicates image is not MEF'
    return int(num)
#---------------------------------------------------------------------------

def mefCompare_sciext(num,header=None):
    """Check to see if file is multi-extension fits (mef)
    @param num
    @type num: number to compare with number of sci ext.
    @param header
    @type header: pyfits Header object
    """  
    num2 = header['NSCIEXT']
    if num2 != num:
        raise 'CRITICAL: Science Extension failed to match'
        
          
        
        
    
        
