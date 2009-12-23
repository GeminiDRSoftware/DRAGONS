import pyfits as pf
import numpy as np
import re
import glob
import time
import os

def reset_nans(im):
    """
     Reset pixels with Nan value to
     99 (arbitrarily). All other pixels to zero.
    """
    bp = im.copy()
    bp[np.isfinite(im)]=0
    gnan = np.where(np.isnan(im))
    bp[gnan]=99.
    return bp

def restore_nans(im,bp):
    """
      Restore those pixels with values > 50 to nans.
      This procedure works in conjunctio with reset_nans.
    """
    im[bp>5.] = np.nan
    return im
   
def check_dir(dirname,flag):
    """
      Check that directory name have ending '/',
      exist and can be accessed with 'flag' access.
    """
    dirname = os.path.join(dirname.strip(),'')

    if dirname == '': return dirname

    # Verify that directory exist
    if (not os.access(dirname,os.F_OK)):
       errs = '"'+dirname+'"'+' does not exist.'
       raise IOError(errs)
    if (not os.access(dirname,flag)):
       acc = {os.R_OK:'read',os.W_OK:'write'}[flag]
       errs = '"'+dirname+'"'+' does not have '+acc+' access.'
       raise IOError(errs)
    return dirname

def gstat(files,path):
    """
      List information about each file in the input
      list. FILE GCALSHUT CRMODE TYPE MEDR MEDB
      where medr and medb are the median value of the
      red and blue frame respectively.
      
    """

    if (path != '' and path[-1] != '/'):
       path += '/'

    print 'FILE GCALSHUT CRMODE TYPE MEDR MEDB'
    for f in files:
        un = pf.open(path+f)
	if (un[0].header).get('instrume') != 'NICI': 
            print f,'is not NICI file.'
            un.close()
            continue
	if (un[1].header).get('FILTER_R'): 
            exr=1; exb=2
        else:
            exr=2; exb=1
        gcal=(un[0].header).get('GCALSHUT')
        otype=(un[0].header).get('OBSTYPE')
        crmode=(un[0].header).get('crmode')
        medb = np.median(un[exb].data,axis=None)
        medr = np.median(un[exr].data,axis=None)
        print f,gcal,otype,crmode ,'(%.2f)' % medr, '(%.2f)' % medb
        un.close()
        #h=iraf.images.hselect(f+'[0]','$I,GCALSHUT,OBSTYPE','yes',Stdout=1)
        #iraf.images.imstat(f+'[1]',format='no',Stdout=1)
        #iraf.images.imstat(f+'[2]',format='no',Stdout=1)

    return 
def create_list(pattern, file=None):
    """
      Create a gemini file name list. The pattern needs to be of
      the form S20090312S+N-M+(.fits), and/or S20090312S+N+(.fits) and/or
      S20090312S0013(.fits). The extension '.fits' is optional.
      file: Creates a filelist is a file is specified.

      Returns a list and a file if one is giving.

    """
    list = []
    files = []
    list = pattern.split(',')
    # split pattern  root+n-m+ into a list of 
    # [root+str(seq) for seq in range(n,m)]
    rex = r'(?P<root>\w*)\+(?P<r1>\d+)-?(?P<r2>\d*)(\D*)'
    saveroot = ''
    for fpat in list:
        m = re.search(rex, fpat)
        root = m.group('root').strip()
        if len(root) > 1:
            if root[-1] == '+':
                root = root[:-1]
        a = int(m.group('r1'))
        b = m.group('r2')
        if root != '':
            saveroot = root
        if b != '':              # we have a range
            b = int(b)
            for n in range(a,b+1):
                files.append(root + str('%.4d'%n) + '.fits')
        else:
            if saveroot == '':
                print "ERROR: The first element in the input list should \
                       have a root string"
                break
            files.append(saveroot + str('%.4d'%a) + '.fits')
               
    return files

def getFileList(input):
    """
    input: (string) filenames separated by commas or an @file.
           If a filename contains a Unix wildcard (?,*,[-])
           then it will attemp to generate the corresponding
           list of files that matches the template.

    """
    filenames=[]
    if 'str' in str(type(input)):
        if input[0] == '@' and input.find (',') < 0:
            fd = open(input[1:])
            flist = fd.readlines()
            fd.close()
        else:
            flist = input.split(',')
    else:
        if len(input) == 1:
           list = getFileList(input[0])
           filenames += list
           return filenames
        else:   
           flist = input

    for line in flist:
        line = line.strip()
        if len(re.findall(r'\w?\+\d*-\d*\+',line)):     # match 'root+N-N+'
            filenames += create_list(line)
        elif len(re.findall (r'[?,*,-]',line)):
           # We got a wilcard as a filename
           line = glob.glob(line)
           filenames += line
        elif '@' in line:
            list = getFileList(line)
            filenames += list
        else:
            if (len(line.strip()) == 0):
                continue
            filenames.append (line)

    return filenames


def gen_list(wildname, keyname, value):
    """
      list = gen_list(wildname, keyname, value):
      arguments:
      wildname: wild card string: eg: *.fits
      keyname: keyword names
      value: string contained in keyname value
      returns: filename list.      

      Generate a list of filenames that matches the 'value' 
      in a keyword. The keyword is taken from the global extension,
      i.e. the PHU
    """
    import glob

    file_list = []
    flis = glob.glob(wildname)
    for f in flis:
        fits=pf.open(f)
        objval=(fits[0].header).get(keyname)
        if (objval == None):
           continue
        if (value.upper() in objval.upper()):
           file_list.append(f)
        fits.close()
    return np.sort(file_list)


def medbin(im, dimx, dimy):

   naxis = np.size(np.shape(im))
   if (naxis == 2):
      dim = [dimx, dimy]
      sz = np.shape(im)
      bx = sz[1] / dim[1]
      by = sz[0] / dim[0]

      index = np.arange(sz[1]*sz[0]).reshape(sz[1],sz[0])
      iy = index % sz[0]
      ix = index / sz[0]
      ipix = ((index % by) + ((ix) % bx) * by)

      iy = iy / by
      ix = ix / bx
      ibox = iy + ix * dim[1]

      v = np.zeros([bx * by, dim[0] * dim[1]], np.float)
      v[ipix,ibox] = im

      #IDL:v = median(v, dim=2, even=True)
      v = np.median(v,axis=0)   # The 1st python dimension is idl's 2nd.

      out = np.zeros([dim[1], dim[0]], np.float)
      sz = np.shape(out)
      out = v.reshape(sz[1],sz[0])

      return out


   if (naxis == 1):
      out = np.zeros([dimx], np.float)
      for i in np.arange(0, (dimx - 1)+(1)):
         out[i] = np.median(im[i*(sz[1]/dimx) : ((i+1)*(sz[1]/dimx)-1)+1])
      return outflat_file

def rebin_simple( a, newshape ):
    '''
      Rebin an array to a new shape.
    '''
    assert len(a.shape) == len(newshape)

    slices = [slice(0,old, np.float(old)/new) for old,new in zip(a.shape,newshape)]
    coordinates = np.mgrid[slices]
    indices = coordinates.astype('i')  #choose the biggest smaller integer index
    return a[tuple(indices)]

def rebin(a, *args):
    '''rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4 rows
    can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    example usages:
    >>> a=rand(6,4); b=rebin(a,3,2)
    >>> a=rand(6); b=rebin(a,2)
    '''
    lenShape = np.size(np.shape(a))
    factor = np.asarray(np.shape(a))/np.asarray(args)
    evList = ['a.reshape('] + \
             ['args[%d],factor[%d],'%(i,i) for i in range(lenShape)] + \
             [')'] + ['.sum(%d)'%(i+1) for i in range(lenShape)] + \
             ['/factor[%d]'%i for i in range(lenShape)]
    #print ''.join(evList)
    return eval(''.join(evList))

def parangle(ha,dec,lat):
    """
    Return the parallactic angle of a source in degrees.

    HA - the hour angle of the source in decimal hours; a scalar or vector.
    DEC - the declination of the source in decimal degrees; a scalar or
            vector.
    LAT - The latitude of the telescope; a scalar.
    """
    from numpy import pi,sin,cos,tan 

    har = np.radians(15*ha)
    decr = np.radians(dec)
    latr = np.radians(lat)

    ac2 = np.arctan2( -sin(har), \
                 cos(decr)*tan(latr) - sin(decr)*cos(har))

    return -np.degrees(ac2)

def dmstod(dms):
    """ 
    return decimal degrees from +/-dd:mm:ss.dd
    """


    dms=dms.strip()
    if dms[0] == '-':
       sign = -1.
    else:
       sign = 1.
    dar = np.asfarray(dms.split(':'))
    dar[0] = sign*dar[0]

    return sign*(dar[0] + dar[1]/60. + dar[2]/3600.)

def dtodms(deg):
    """
    Turns degrees (a floating) into +/-dd:mm:ss.ddd
    """

   # Separate out the sign.
    if deg < 0.0:
        sign = '-'
        deg = -deg
    else:
        sign = '+'

    # Convert to an integer to avoid inconsistent rounding later.
    # The least significant digit of the result corresponds to
    # centiseconds so multiply the number accordingly.  Note that
    # int rounds towards zero.
    csec = int(deg * 60.0 * 60.0 * 100.0 + 0.5)

    # Split up into the four components.  divmod calculates
    # quotient and remainder simultaneously.
    sec, csec = divmod(csec, 100)
    min, sec = divmod(sec, 60)
    deg, min = divmod(min, 60)

    # Convert the four components and sign into a string.  The
    # % operator on strings works like C's sprintf function.
    return '%s%d:%02d:%2d.%02d' % (sign, deg, min, sec, csec)

def printlog_open(logfile):
    """
    Open logfile for appending
    BETTER TO USE THE Python Logging facility
    """
    lg = open(logfile,'a')
    return lg

def print_error(lg,error):
    pass
    # need to write a class

def nici_noise(imc):
    import copy
    """
     A little function that subtracts the nici readout
     pattern noise every 16 pixels
    """
    im = copy.deepcopy(imc)
    line= np.zeros(16)
    for q in range(2):
        for i in range(128):
            tmp=im[i*8:i*8+8, q*512:q*512+512].ravel()
            for j in range(16): 
                line[j] = np.median(tmp[np.arange(256)*16+j])
            line -= np.median(line)
            for j in range(16):  
               tmp[np.arange(256)*16+j] -= line[j]
            im[i*8:i*8+8, q*512:q*512+512] = tmp.reshape(8,512)

    return im

def fits_utc():
   """
   Return a UTC string in FITS format:
   YYYY-MM-DDThh:mm:ss
   """

   gmt = time.gmtime()
   time.asctime(gmt)
   fitsT = '%d-%02d-%02dT%02d:%02d:%02d' % gmt[:6]

   return fitsT

def order_wcs(header):
   """
   Order the WSC information in the FITS header according to
   RADECSYS, CTYPEn,CRPIXn,CRVALn and CDn_m for n,m [1:3].
   """
   nwcs=[]
   hcc = header.ascard
   for pat in ['RADECSYS','CTYPE.','CRPIX.','CRVAL.', 'CD._.']:
       for h in hcc.filterList(pat):
           nwcs.append(h)
           kw = str(h).split('=')[0].strip()
           header.__delitem__(kw)
   for cc in nwcs:
        hcc.append(cc)


def parallacticAngle (filelist, fdir='',extn=1):
    """
      Calculates the parallactic angle from a FITS header WCS cd matrix.
      Reads the first FITS extension to get the WCS values  ([1])
      
      @filelist:  A list of FITS files
      @type filelist: Python list
      @fdir:      Directory pathname where FITS files are located.
      @type fdir: String (Default value is '')  
      @extn:      FITS extension number to open
      @type extn: int (Default value is 1)

      Returns: A list of corresponding parallactic angles.
    """
    # Get information from the EHU to calculate the Parallactic angle
    #
    fdir = os.path.join(fdir,'')
    radeg = 180/np.pi
    pa = []
    for fi in filelist:
        fi = fdir+fi
        hdr = pf.getheader(fi,extn)
        cd11=hdr['cd1_1']
        cd12=hdr['cd1_2']
        cd21=hdr['cd2_1']
        cd22=hdr['cd2_2']
        dd = 180 - np.arctan2(sum([cd11,cd22]),sum([cd21,-cd12]))*radeg
        pa.append(dd)

    return pa

