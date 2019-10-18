"""numdisplay: Package for displaying numpy arrays in IRAF-compatible
                image display tool such as DS9 or XIMTOOL.

    This package provides several methods for controlling the display
    of the numpy array; namely,

        open(imtdev=None)::
            Open the default display device or the device specified
            in 'imtdev', such as 'inet:5137' or 'fifo:/dev/imt1o'.

        close()::
            Close the display device defined by 'imtdev'. This must
            be done before resetting the display buffer to a new size.

        display(pix, name=None, bufname=None, z1=None, z2=None, quiet=False, transform=None, scale=None, offset=None, frame=None)::
            Display the scaled array in display tool (ds9/ximtool/...).

        readcursor(sample=0)::
            Return a single cursor position from the image display.
            By default, this operation will wait for a keystroke before
            returning the cursor position. If 'sample' is set to 1,
            then it will NOT wait to read the cursor.
            This will return a string containing: x,y,frame and key.

        help()::
            print Version ID and this message.

    Notes
    -----
    Displaying a numpy array object involves:
        1.  Opening the connection to a usable display tool (such as DS9).

        2.  Setting the display parameters for the array, such as min and
            max array value to be used for min and max grey scale level, along
            with any offset, scale factor and/or transformation function to be
            applied to the array.
        3.  Applying any transformation to the input array.  This transformation
            could be a simple numpy ufunc or a user-defined function that
            returns a modified array.

        4.  Building the byte-scaled version of the transformed array and
            sending it to the display tool.  The image sent to the display
            device will be trimmed to fit the image buffer defined by the
            'imtdev' device from the 'imtoolrc' or the 'stdimage'
            variable under IRAF. If the image is smaller than the buffer,
            it will be centered in the display device.

            All pixel positions reported back will be relative to the
            full image size.

    Examples
    --------

    The user starts with a 1024x1024 array in the variable 'fdata'.
    This array has min pixel value of -157.04 and a max pixel value
    of 111292.02.  The display tool DS9 has already been started from
    the host level and awaits the array for display.  Displaying the
    array requires::

        >>> import numdisplay
        >>> numdisplay.display(fdata)

    If there is a problem connecting to the DS9 application, the connection
    can be manually started using::

        >>> numdisplay.open()

    To bring out the fainter features, an offset value of 158 can be added
    to the array to allow a 'log' scaling can be applied to the array values
    using::

        >>> numdisplay.display(fdata,transform=numpy.log,offset=158.0)

    To redisplay the image with default full-range scaling::

        >>> numdisplay.display(fdata)

    To redisplay using the IRAF display zscale algorithm, and with a contrast
    value steeper than the default value of 0.25::

        >>> numdisplay.display(fdata, zscale=True, contrast=0.5)


"""
from __future__ import absolute_import, division, print_function # confidence medium

from .version import *

import numpy as n
import math, string
from . import displaydev
from . import zscale as _zscale

try:
    import geotrans
except ImportError:
    geotrans = None


class NumDisplay(object):
    """ Class to manage the attributes and methods necessary for displaying
        the array in the image display tool.

        This class contains the methods:
            open(imtdev=None):

            close():

            set(z1=None,z2=None,scale=None,factor=None,frame=None):
            reset(names=None)

            display(pix, name=None, bufname=None):

            readcursor():

    """
    def __init__(self):
        self.frame = 1

        # Attributes used to scale image nearly arbitrarily
        # transform: name of Python function to operate on array
        #           default is to do nothing (apply self._noTransform)
        # scale : multiplicative factor to apply to input array
        # offset: additive factor to apply to input array

        self.transform = self._noTransform
        self.scale = None
        self.offset = None

        # default values for attributes used to determine pixel range values
        self.zscale = False
        self.stepline = 6
        self.contrast = 1    # Not implemented yet!
        self.nlines = 256    # Not implemented yet!

        # If zrange != 0, use user-specified min/max values
        self.zrange = 0  # 0 == False

        # Scale the image based on input pixel range...
        self.z1 = None
        self.z2 = None

        self.name = None
        self.view = displaydev._display
        self.handle = self.view.getHandle()

    def open(self,imtdev=None):
        """ Open a display device. """
        self.view.open(imtdev=imtdev)

    def close(self):
        """ Close the display device entry."""
        self.view.close()

    def set(self,frame=None,z1=None,z2=None,contrast=None,transform=None,scale=None,offset=None):

        """ Allows user to set multiple parameters at one time. """

        self.frame = frame
        self.zrange = 0  # 0 == False

        if contrast != None:
            self.contrast = contrast

        if z1 != None:
            self.z1 = z1
            self.zrange = 1  # 1 == True

        if z2 != None:
            self.z2 = z2
            self.zrange = 1  # 1 == True


        if transform:
            self.transform = transform

        if scale != None:
            self.scale = scale

        if offset is not None:
            self.offset = offset

    def reset(self, names=None):
        """Reset specific attributes, or all by default.

        Parameters
        ----------
        names : string or list of strings
            names of attributes to be reset, separated by commas
            or spaces; the default is to reset all attributes to None

        """

        if names:
            if isinstance(names, str):
                names = names.replace(",", " ")
                names = names.split()
            for name in names:
                if name == "transform":
                    self.transform = self._noTransform
                elif name == "zrange":
                    self.zrange = 0     # False
                else:
                    self.__setattr__(name, None)
            if "z1" in names or "z2" in names:
                self.zrange = 0         # False
        else:
            self.contrast = None
            self.z1 = None
            self.z2 = None
            self.transform = self._noTransform
            self.scale = None
            self.offset = None
            self.zrange = 0             # False

    def _noTransform(self, image):
        """ Applies NO transformation to image. Returns original.
            This will be the default operation when None is specified by user.
        """
        return image

    def _bscaleImage(self, image):
        """
        This function converts the input image into a byte-array
         with the z1/z2 values being mapped from 1 - 200.
         It also returns a status message to report on success or failure.

        """
        _pmin = 1.
        _pmax = 200.
        _ny,_nx = image.shape

        bimage = n.zeros((_ny,_nx),dtype=n.uint8)
        iz1 = self.z1
        iz2 = self.z2

        if iz2 == iz1:
            status = "Image scaled to all one pixel value!"
            return bimage
        else:
            scale =  (_pmax - _pmin) / (iz2 - iz1)

        # Now we can scale the pixels using a linear scale only (for now)
        # Scale the pixel values:  iz1 --> _pmin, iz2 --> _pmax
        bimage = (image - iz1) * scale + _pmin
        bimage = n.clip(bimage,_pmin,_pmax)
        bimage = n.array(bimage,dtype=n.uint8)

        status = 'Image scaled to Z1: '+repr(iz1)+' Z2: '+repr(iz2)+'...'
        return bimage


    def _fbclipImage(self,pix,fbwidth,fbheight):

        # Get the image parameters
        _ny,_nx = pix.shape

        if _nx > fbwidth or _ny > fbheight:

            # Compute the starting pixel of the image section to be displayed.
            _lx = (_nx // 2) - (fbwidth // 2)
            _ly = (_ny // 2) - (fbheight // 2)
            # We need to determine the region of the image to be put in frame
            _nx = min(_nx,fbwidth)
            _ny = min(_ny,fbheight)

        else:
            _lx = 0
            _ly = 0
        # Determine pixel range for image (sub)section
        # Insure it does not go beyond actual image array data bounds
        _xstart = max(_lx,0)
        _xend = max( (_lx + _nx),_nx)
        _ystart = max(_ly, 0)
        _yend = max( (_ly + _ny), _ny)

        # Return bytescaled, frame-buffer trimmed image
        if (_xstart == 0 and _xend == pix.shape[0]) and (_ystart == 0 and _yend == pix.shape[1]):
            return self._bscaleImage(pix)
        else:
            return self._bscaleImage(pix[_ystart:_yend,_xstart:_xend])

    def _transformImage(self, pix):
        """ Apply user-specified scaling to the input array. """

        if isinstance(pix,n.ndarray):

            if self.zrange:
                zpix = pix.copy()
                zpix = n.clip(pix,self.z1,self.z2)
            else:
                zpix = pix
        else:
            zpix = pix
        # Now, what kind of multiplicative scaling should be applied
        if self.scale:
            # Apply any additive offset to array
            if self.offset is not None:
                return self.transform( (zpix+self.offset)*self.scale)
            else:
                return self.transform( zpix*self.scale)
        else:
            if self.offset is not None:
                return self.transform (zpix + self.offset)
            else:
                return self.transform(zpix)

    def display(self, pix, name=None, bufname=None, z1=None, z2=None,
             transform=None, zscale=False, contrast=0.25, scale=None,
             offset=None, frame=None,quiet=False):

        """ Displays byte-scaled (UInt8) n to XIMTOOL device.
        This method uses the IIS protocol for displaying the data
        to the image display device, which requires the data to be
        byte-scaled.

        If input is not byte-scaled, it will perform scaling using
        set values/defaults.

        Parameters
        ----------
        name : str
            optional name to pass along for identifying array

        bufname : str
            name of buffer to use for displaying array (such as 'imt512').
            Other valid values include::

                'iraf': look for 'stdimage' and use that buffer or default to 'imt1024' [1024x1024 buffer]
                None  : ignore 'stdimage' and automatically select a buffer matched to the size of the image.

        z1,z2 : float
            minimum/maximum pixel value to display. Not specifying values will default
            to the full range values of the input array.

        transform : function
            Python function to apply to array (function)

        zscale : bool
            Specify whether or not to use an algorithm like that in the IRAF
            display task. If zscale=True, any z1 and z2 set in the call to display
            are ignored.  Using zscale=True invalidates any transform
            specified in the call.

        contrast : float (Default=0.25)
            Same as the *contrast* parameter in the IRAF *display* task.
            Only applies if zscale=True. Higher contrast values make z1 and
            z2 closer together, while lower values give a gentler (wider) range.

        scale : float/int
            multiplicative scale factor to apply to array. The value of this
            parameter remains persistent, so to reset it you must specify
            scale=1 in the display call.

        offset : float/int
            additive factor to apply to array before scaling. This value is
            persistent, so to reset it you have to set it to 0.

        frame : int
            image buffer frame number in which to display array

        quiet : bool (Default: False)
            if True, this parameter will turn off all status messages

        Notes
        ------
        The display parameters set here will ONLY apply to the display
        of the current array.

        """

        #Ensure that the input array 'pix' is a numpy array
        pix = n.array(pix)
        self.z1 = z1
        self.z2 = z2

        # If any of the display parameters are specified here, apply them
        #if z1 or z2 or transform or scale or offset or frame:
        # If zscale=True (like IRAF's display) selected, calculate z1 and z2 from
        # the data, and clear any transform specified in the call
        # Offset and scale arew applied to the data and z1,z2, so they have no effect
        # on the display
        if zscale:
            if transform != None:
                if not quiet:
                    print("transform disallowed when zscale=True")
                transform = None

            z1, z2 = _zscale.zscale(pix, contrast=contrast)

        self.set(frame=frame, z1=z1, z2=z2,
                transform=transform, scale=scale, offset=offset)

        # Initialize the display device
        if not self.view._display or self.view.checkDisplay() is False:
            self.open()
        _d = self.view._display
        self.handle = _d.getHandle()

        # If no user specified values are provided, interrogate the array itself
        # for the full range of pixel values
        if self.z1 == None:
            self.z1 = n.minimum.reduce(n.ravel(pix))
        if self.z2 == None:
            self.z2 = n.maximum.reduce(n.ravel(pix))

        # If the user has not selected a specific buffer for the display,
        # select and set the frame buffer size based on input image size.
        if bufname == 'iraf':
            useiraf = True
            bufname = None
        else:
            useiraf = False

        if bufname != None:
            _d.setFBconfig(None,bufname=bufname)
        else:
            _ny,_nx = pix.shape
            _d.selectFB(_nx,_ny,reset=1,useiraf=useiraf)

        # Initialize the specified frame buffer
        _d.setFrame(self.frame)
        _d.eraseFrame()

        # Apply user specified scaling to image, returns original
        # if none are specified.

        bpix = self._transformImage(pix)

        # Recompute the pixel range of (possibly) transformed array
        _z1 = self._transformImage(self.z1)
        _z2 = self._transformImage(self.z2)

        # If there was a problem in the transformation, then restore the original
        # array as the one to be displayed, even though it may not be ideal.
        if _z1 == _z2:
            if not quiet:
                print('Error encountered during transformation. No transformation applied...')
            bpix = pix
            self.z1 = n.minimum.reduce(n.ravel(bpix))
            self.z2 = n.maximum.reduce(n.ravel(bpix))
            # Failsafe in case input image is flat:
            if self.z1 == self.z2:
                self.z1 -= 1.
                self.z2 += 1.
        else:
            # Reset z1/z2 values now so that image gets displayed with
            # correct range.  Also, when displaying transformed images
            # this allows the input pixel value to be displayed, rather
            # than the transformed pixel value.
            self.z1 = _z1
            self.z2 = _z2

        _wcsinfo = displaydev.ImageWCS(bpix,z1=self.z1,z2=self.z2,name=name)
        if not quiet:
            print('Image displayed with Z1: ',self.z1,' Z2:',self.z2)

        bpix = self._fbclipImage(bpix,_d.fbwidth,_d.fbheight)

        # Update the WCS to match the frame buffer being used.
        _d.syncWCS(_wcsinfo)

        # write out WCS to frame buffer, then erase buffer
        _d.writeWCS(_wcsinfo)

        # Now, send the trimmed image (section) to the display device
        _d.writeImage(bpix,_wcsinfo)
        #displaydev.close()

    def readcursor(self,sample=0):
        """ Return the cursor position from the image display. """
        return self.view.readCursor(sample=sample)

    def getHandle(self):
        return self.handle

    def checkDisplay(self):
        return self.view.checkDisplay()

# Help facility
def help():
    """ Print out doc string with syntax and example. """
    print('numdisplay --- Version ',__version__)
    print(__doc__)


view = NumDisplay()


# create aliases for PyDisplay methods
open = view.open
close = view.close

set = view.set
display = view.display
readcursor = view.readcursor
getHandle = view.getHandle
checkDisplay = view.checkDisplay

def sample() :
    '''stuff a sample image into the display

    use this to see that numdisplay is able to speak to the display program
    (ds9 or ximtool)
'''
    #
    numpy = n

    # an array of values 0..99
    a1 = numpy.arange(100)

    # an array going from 200..0 by 2
    a2 = numpy.arange(200,0,-2)

    # an empty array the size of the image I want
    b = numpy.zeros( 100 * 100 )

    # make the array square
    b.shape = ( 100, 100 )

    # copy a1 into the first 50 rows
    for x in range(0, 50) :
        b[x] = a1

    # copy a2 into the second 50 rows
    for x in range( 50, 100 ) :
        b[x] = a2

    # numdisplay.display(b)
    display(b)

    print("The first 50 rows are ascending brightness left  to right")
    print("The next  50 rows are ascending brightness right to left, but st")
    print("REMEMBER THAT 0,0 IS BOTTOM LEFT")
