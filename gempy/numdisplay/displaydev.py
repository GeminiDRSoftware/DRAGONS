"""displaydev.py: Interact with IRAF-compatible image display

Modeled after the NOAO Client Display Library (CDL)

Public functions:

readCursor(sample=0)
        Read image cursor position

open(imtdev=None)
        Open a connection to the display server.  This is called
        automatically by readCursor if the display has not already been
        opened, so it is not generally necessary for users to call it.

        See the open doc string for info on the imtdev argument, which
        allows various forms of socket and network connections.

close()
        Close the active display server.  Called automatically on exit.

Various classes are defined for the different connections (ImageDisplay,
ImageDisplayProxy, UnixImageDisplay, InetImageDisplay, FifoImageDisplay).
They should generally be created using the _open factory function.
This could be used to maintain references to multiple display servers.

Ultimately more functionality may be added to make this a complete
replacement for CDL.

$Id$

"""

import os, socket, struct

import numpy as n
from . import imconfig

try:
    # Only try to import this on Unix-compatible systems,
    # since Win platforms does not have the 'fcntl' library.
    import fcntl

    # FCNTL is deprecated in Python 2.2
    if hasattr(fcntl,'F_SETFL'):
        FCNTL = fcntl
    else:
        import FCNTL
except ImportError as error:
    fcntl = None
    # set a default that will be compatible with Win platform
    os.O_NDELAY = 0

try:
    from stsci.tools import fileutil
except:
    fileutil=None

try:
    SOCKTYPE = socket.AF_UNIX
except AttributeError as error:
    SOCKTYPE = socket.AF_INET

SZ_BLOCK = 16384

_default_imtdev = ("unix:/tmp/.IMT%d", "fifo:/dev/imt1i:/dev/imt1o","inet:5137")
_default_fbconfig = 3

class ImageWCS:
    _W_UNITARY = 0
    _W_LINEAR = 1
    _W_LOG = 2
    _W_USER = 3

    def __init__(self,pix,name=None,title=None,z1=None,z2=None):
        # pix must be a numpy array...
        self.a = 1.0
        self.b = self.c = 0.
        self.d = -1.0
        _shape = pix.shape
        # Start assuming full image can fit in frame buffer
        self.tx = int(_shape[1] / 2. + 0.5)
        self.ty = int(_shape[0] / 2. + 0.5)
        self.dtx = _shape[1] / 2.
        self.dty = _shape[0] / 2.

        # Determine full range of pixel values for image
        if not z1:
            self.z1 = n.minimum.reduce(n.ravel(pix))
        else:
            self.z1 = z1

        if not z2:
            self.z2 = n.maximum.reduce(n.ravel(pix))
        else:
            self.z2 = z2

        self.zt = self._W_LINEAR

        if not name:
            self.name = 'Image'
        else:
            self.name = name
        self.title = title

        self.ny,self.nx = pix.shape
        self.full_ny, self.full_nx = pix.shape

    def update(self,wcsstr):
        # This routine will accept output from readWCS and
        # update the WCS attributes with the values
        if wcsstr != None:
            _wcs = wcsstr.split()
            self.name = _wcs[0]
            self.a = float(_wcs[1])
            self.b = float(_wcs[2])
            self.c = float(_wcs[3])
            self.d = float(_wcs[4])
            self.tx = float(_wcs[5])
            self.ty = float(_wcs[6])
            self.z1 = float(_wcs[7])
            self.z2 = float(_wcs[8])
            self.zt = int(_wcs[9])


    def __str__(self):
        # This method can be used for obtaining the string
        # necessary for setting the WCS of the frame buffer
        # Start by insuring that the image name if not set is
        # passed as the string 'None' at least

        _name = self.name
        if self.title != None:
            _name = _name+'-'+repr(self.title)
        _name =_name+'\n'

        _str = _name+str(self.a)+' '+str(self.b)+' '+str(self.c)+' '+str(self.d)
        _str = _str+' '+str(self.tx)+' '+str(self.ty)
        _str = _str+' '+str(self.z1)+' '+str(self.z2)+' '+str(self.zt)+'\n'

        return(_str)


def _open(imtdev=None):

    """Open connection to the image display server

    This is a factory function that returns an instance of the ImageDisplay
    class for the specified imtdev.  The default connection if no imtdev is
    specified is given in the environment variable IMTDEV (if defined) or
    is "unix:/tmp/.IMT%d".  Failing that, a connection is attempted on the
    /dev/imt1[io] named fifo pipes.

    The syntax for the imtdev argument is <domain>:<address> where <domain>
    is one of "inet" (internet tcp/ip socket), "unix" (unix domain socket)
    or "fifo" (named pipe).  The form of the address depends upon the
    domain, as illustrated in the examples below.
    ::

     inet:5137                  Server connection to port 5137 on the local
                                host.  For a client, a connection to the
                                given port on the local host.

     inet:5137:foo.bar.edu      Client connection to port 5137 on internet
                                host foo.bar.edu.  The dotted form of address
                                may also be used.

     unix:/tmp/.IMT212          Unix domain socket with the given pathname
                                IPC method, local host only.

     fifo:/dev/imt1i:/dev/imt1o FIFO or named pipe with the given pathname.
                                IPC method, local host only.  Two pathnames
                                are required, one for input and one for
                                output, since FIFOs are not bidirectional.
                                For a client the first fifo listed will be
                                the client's input fifo; for a server the
                                first fifo will be the server's output fifo.
                                This allows the same address to be used for
                                both the client and the server, as for the
                                other domains.

    The address field may contain one or more "%d" fields.  If present, the
    user's UID will be substituted (e.g. "unix:/tmp/.IMT%d").
    """

    if not imtdev:
        # try defaults
        defaults = list(_default_imtdev)
        if 'IMTDEV' in os.environ:
            defaults.insert(0,os.environ['IMTDEV'])
        for imtdev in defaults:
            try:
                return _open(imtdev)
            except:
                pass
        raise OSError("Cannot attach to display program. Verify that one is running...")
    # substitute user id in name (multiple times) if necessary
    nd = len(imtdev.split("%d"))
    if nd > 1:
        dev = imtdev % ((abs(os.getpid()),)*(nd-1))
    else:
        dev = imtdev
    fields = dev.split(":")
    domain = fields[0]
    if domain == "unix" and len(fields) == 2:
        return UnixImageDisplay(fields[1])
    elif domain == "fifo" and len(fields) == 3:
        return FifoImageDisplay(fields[1],fields[2])
    elif domain == "inet" and (2 <= len(fields) <= 3):
        try:
            port = int(fields[1])
            if len(fields) == 3:
                hostname = fields[2]
            else:
                hostname = None
            return InetImageDisplay(port, hostname)
        except ValueError:
            pass
    raise ValueError("Illegal image device specification `%s'"
                                    % imtdev)


class ImageDisplay:

    """Interface to IRAF-compatible image display"""

    # constants for IIS Protocol header packets
    _IIS_READ =   32768   # octal 0100000
    _IIS_WRITE =  131072  # octal 0400000
    _COMMAND =    32768   # octal 0100000
    _PACKED =     16384   # octal 0040000
    _IMC_SAMPLE = 16384   # octal 0040000

    _MEMORY = 1     # octal 01
    _LUT    = 2     # octal 02
    _FEEDBACK = 5   # octal 05
    _IMCURSOR = 16  # octal 020
    _WCS = 17       # octal 021

    _SZ_IMCURVAL = 160
    _SZ_WCSBUF = 320

    def __init__(self):
        # Flag indicating that readCursor request is active.
        # This is used to handle interruption of readCursor before
        # read is complete.  Without this kluge, ^C interrupts
        # leave image display in a bad state.
        self._inCursorMode = 0

        # Add hooks here for managing frame configuration
        self.fbdict = imconfig.loadImtoolrc()

        self.frame = 1

        self.fbname = None

        _fbconfig = self.getDefaultFBConfig()

        self.fbconfig = _fbconfig
        self.fbwidth = self.fbdict[self.fbconfig]['width']
        self.fbheight = self.fbdict[self.fbconfig]['height']


    def getDefaultFBConfig(self):
        try:
            # Try to use the IRAF 'stdimage' value as the default
            # fbconfig number, if it exists
            if fileutil is not None:
                self.fbname = fileutil.envget('stdimage')
            else:
                if 'stdimage' in os.environ:
                    self.fbname = os.environ['stdimage']

            if self.fbname is not None:
                # Search through all IMTOOLRC entries to find a match
                _fbconfig = self.getConfigno(self.fbname)
            else:
                _fbconfig = _default_fbconfig
        except:
            _fbconfig = _default_fbconfig

        return _fbconfig

    def readCursor(self,sample=0):

        """Read image cursor value for this image display

        Return immediately if sample is true, or wait for keystroke
        if sample is false (default).  Returns a string with
        x, y, frame, and key.
        """

        if not self._inCursorMode:
            opcode = self._IIS_READ
            if sample:
                opcode |= self._IMC_SAMPLE
            self._writeHeader(opcode, self._IMCURSOR, 0, 0, 0, 0, 0)
            self._inCursorMode = 1
        s = self._read(self._SZ_IMCURVAL)
        self._inCursorMode = 0
        # only part up to newline is real data
        try:
            coo = s.split("\n")[0]  # works in Py2
        except TypeError:
            coo = s.decode("utf-8", "ignore").split("\n")[0]  # works in Py3
        return coo
        #return s.split("\n")[0]

    def getConfigno(self,stdname):

        """ Determine which config number matches
        specified frame buffer name.
        """
        _fbconfig = None
        # Search through all IMTOOLRC entries to find a match
        for fb in self.fbdict:
            if self.fbdict[fb]['name'].find(stdname.strip()) > 0:
                _fbconfig = int(fb)
                break
        if not _fbconfig:
            # If no matching configuration found,
            # default to 'imt1024'
            _fbconfig = _default_fbconfig

        return _fbconfig

    def selectFB(self,nx,ny,reset=None,useiraf=True):
        """ Select the frame buffer that best matches the input image size."""
        newfb = None
        _tmin = 100000

        #if iraffunctions:
        if self.fbname is not None and useiraf:
            # Use the STDIMAGE variable defined in the IRAF session...
            newfb = self.getDefaultFBConfig()
        else:
            # Otherwise, fall back to finding the buffer with the
            # size closest to the image's size.
            for fb in self.fbdict:
                _fbw = self.fbdict[fb]['width']
                _fbh = self.fbdict[fb]['height']
                if nx == _fbw and ny == _fbh:
                    # Found an exact match.
                    newfb = fb
                    break
                elif _fbw > nx and _fbh > ny:
                    # No exact match, so look for match with smallest padding.
                    _edges = _fbw - nx + _fbh - ny
                    if _edges < _tmin:
                        _tmin = _edges
                        newfb = fb

        # If no new frame buffer was found that matched better than default...
        if not newfb:
            # use the default (probably 'imt512')
            newfb = self.fbconfig

        # If reset was specified, automatically set new config to this value.
        if reset:
            self.setFBconfig(newfb)

        # At the very least, return the config number found.
        return newfb


    def setFBconfig(self,fbnum,bufname=None):

        """ Set the frame buffer values for the given frame buffer name. """

        if bufname:
            self.fbconfig = self.getConfigno(bufname)
        else:
            self.fbconfig = fbnum

        self.fbwidth = self.fbdict[self.fbconfig]['width']
        self.fbheight = self.fbdict[self.fbconfig]['height']

    def writeData(self,x,y,pix):

        """ Writes out image data to x,y position in active frame. """

        opcode = self._IIS_WRITE | self._PACKED
        frame = 1 << (self.frame-1)
        nbytes = pix.size * pix.itemsize
        self._writeHeader(opcode,self._MEMORY, -nbytes, x, y, frame, 0)

        status = self._write(pix.tobytes())
        return status

    def readData(self,x,y,pix):

        """ Reads data from x,y position in active frame."""

        opcode = self._IIS_READ | self._PACKED
        nbytes = pix.size * pix.itemsize
        frame = 1 << (self.frame-1)
        self._writeHeader(opcode,self._MEMORY, -nbytes, x, y, frame, 0)

        # Get the pixels now
        return self._read(nbytes)

    def setCursor(self,x,y,wcs):

        """ Moves cursor to specified position in frame. """

        self._writeHeader(self._IIS_WRITE, self._IMCURSOR,0,x,y,wcs,0)

    def setFrame(self,frame_num=1):

        """ Sets the active frame in frame buffer to specified value."""

        code = self._LUT | self._COMMAND
        self._writeHeader(self._IIS_WRITE, code, -1, 0, 0, 0, 0)

        # Update with user specified frame number
        if frame_num:
            self.frame = frame_num

        # Convert to bit-shifted value for IIS stream
        frame = 1 << (self.frame - 1)
        # Write out 2-byte value for frame number
        self._write(struct.pack('H',frame))

    def eraseFrame(self):

        """ Sends commands to erase active frame."""

        opcode = self._IIS_WRITE + self.fbconfig-1

        frame = 1 << (self.frame-1)
        self._writeHeader(opcode, self._FEEDBACK, 0,0,0,frame,0)

    def writeWCS(self,wcsinfo):

        """ Writes out WCS information for frame to display device."""

        _str = str(wcsinfo).rstrip()
        nbytes = len(_str)
        opcode = self._IIS_WRITE | self._PACKED
        frame = 1 << (self.frame-1)
        fbconfig = self.fbconfig - 1

        self._writeHeader(opcode,self._WCS, -nbytes, 0,0, frame, fbconfig)

        status = self._write(_str)

    def readWCS(self,wcsinfo):

        """ Reads WCS information from active frame of display device."""

        frame = 1 << (self.frame-1)

        self._writeHeader(self._IIS_READ, self._WCS, 0,0,0,frame,0)

        wcsinfo.update(self._read(self._SZ_WCSBUF))
        return wcsinfo

    def readInfo(self):
        """Read tx and ty from active frame of display device."""

        frame = 1 << (self.frame-1)

        self._writeHeader(self._IIS_READ, self._WCS, 0,0,0,frame,0)

        wcsstr = self._read(self._SZ_WCSBUF)
        _wcs = wcsstr.split()
        tx = int(round(float(_wcs[5])))
        ty = int(round(float(_wcs[6])))
        # print "debug: ", wcsstr

        return (tx, ty, self.fbwidth, self.fbheight)

    def syncWCS(self,wcsinfo):

        """ Update WCS to match frame buffer being used. """

        # Update WCS information with offsets into frame buffer for image
        wcsinfo.tx = int(((wcsinfo.full_nx + 1.0) / 2.) - ((self.fbwidth) / 2.) + 0.5)
        wcsinfo.ty = int((self.fbheight) + ((wcsinfo.full_ny / 2.) - (self.fbheight / 2.)) + 0.5)

        wcsinfo.nx = min(wcsinfo.full_nx, self.fbwidth)
        wcsinfo.ny = min(wcsinfo.full_ny, self.fbheight)

        # Keep track of the origin of the displayed, trimmed image
        # which fits in the buffer.
        wcsinfo.dtx = int((wcsinfo.nx / 2.) - ((self.fbwidth) / 2.) + 0.5)
        wcsinfo.dty = int((self.fbheight) + ((wcsinfo.ny / 2.) - (self.fbheight / 2.)) + 0.5)

    def writeImage(self,pix,wcsinfo):

        """ Write out image to display device in 32Kb sections."""

        _fbnum = self.fbconfig
        _fbw = self.fbdict[_fbnum]['width']
        _fbh = self.fbdict[_fbnum]['height']
        _nx,_ny = wcsinfo.nx,wcsinfo.ny
        _ty = wcsinfo.dty
        _tx = wcsinfo.dtx

        _nnx = min(_nx,_fbw)
        _nny = min(_ny,_fbh)

        # compute the range in output pixels the input image would cover
        # input image could be smaller than buffer size/output image size.
        _lx = (_fbw // 2) - (_nnx // 2)

        _lper_block = SZ_BLOCK // _fbw
        if _lper_block > 1: _lper_block = 1
        _nblocks = _nny // _lper_block

        # Flip image array so that (0,0) pixel is in upper left
        _fpix = pix[::-1,:]

        # Now, for each block, write out the image section
        if _lper_block == 1:
            # send each line of image to display
            for block in range(int(_nblocks)):
                _ydisp = _fbh - (_ty - block)
                self.writeData(_lx,_ydisp,_fpix[block,:])
        else:
            # display each line segment separately
            for block in range(int(_nblocks)):
                _y0 = block * _lper_block
                _ydisp = _fbh - (_ty - _y0)
                _xper_block = (_nx // (_nx * _lper_block))
                for xblock in range(int(_xper_block)):
                    _x0 = xblock * _xper_block
                    _xend = xblock + 1 * _xper_block
                    if _xend > _nx: _xend = _nx
                    self.writeData(_lx,_ydisp,_fpix[_y0,_x0:_xend])

        #Now pick up last impartial block
        _yend = int(_nblocks) * _lper_block
        if _yend < _ny:
            #self.writeData(_lx,_yend+_ty,_fpix[_yend:0,:])
            self.writeData(_lx,_yend+_ty,_fpix[:_yend,:])


    def _writeHeader(self,tid,subunit,thingct,x,y,z,t):

        """Write request to image display"""

        a = n.array([tid,thingct,subunit,0,x,y,z,t],dtype=int).astype(n.uint16)
        # Compute the checksum
        sum = n.add.reduce(a,dtype=n.uint16)
        sum = 0xffff - (sum & 0xffff)
        a[3] = sum
        self._write(a.tobytes())

    def close(self, os_close=os.close):

        """Close image display connection"""

        try:
            os_close(self._fdin)
        except (OSError, AttributeError):
            pass
        try:
            os_close(self._fdout)
        except (OSError, AttributeError):
            pass

    def getHandle(self):
        return self

    def _read(self, n):
        """Read n bytes from image display and return as string

        Raises IOError on failure.  If a Tkinter widget exists, runs
        a Tk mainloop while waiting for data so that the Tk widgets
        remain responsive.
        """
        try:
            return os.read(self._fdin, n)
        except (EOFError, OSError):
            raise OSError("Error reading from image display")

    def _write(self, s):
        """Write string s to image display

        Raises IOError on failure
        """
        try:
            n = len(s)

            # Python 3 compat
            #  It used to be that in Py2 is wasn't type str, we found cases
            #  where it is, so now, there's a try block too.  Py2 and Py3 
            #  compat.
            if isinstance(s, str):
                try:
                    s = s.encode()
                except UnicodeDecodeError:
                    pass

            while n>0:
                nwritten = self._socket.send(s[-n:])
                n -= nwritten
                if nwritten <= 0:
                    raise OSError("Error writing to image display")
        except OSError:
            raise OSError("Error writing to image display")


class FifoImageDisplay(ImageDisplay):

    """FIFO version of image display"""

    def __init__(self, infile, outfile):
        ImageDisplay.__init__(self)
        try:
            self._fdin = os.open(infile, os.O_RDONLY | os.O_NDELAY)
            if fcntl:
                fcntl.fcntl(self._fdin, FCNTL.F_SETFL, os.O_RDONLY)

            self._fdout = os.open(outfile, os.O_WRONLY | os.O_NDELAY)
            if fcntl:
                fcntl.fcntl(self._fdout, FCNTL.F_SETFL, os.O_WRONLY)
        except OSError as error:
            raise OSError("Cannot open image display (%s)" % (error,))

    def _write(self, s):
        """Write string s to image display

        Raises IOError on failure
        """
        try:
            n = len(s)
            while n>0:
                nwritten = os.write(self._fdout, s[-n:])
                n -= nwritten
                if nwritten <= 0:
                    raise OSError("Error writing to image display")
        except OSError:
            raise OSError("Error writing to image display")

    def __del__(self):
        self.close()

class UnixImageDisplay(ImageDisplay):

    """Unix socket version of image display"""

    def __init__(self, filename, family=SOCKTYPE, type=socket.SOCK_STREAM):
        ImageDisplay.__init__(self)
        try:
            self._socket = socket.socket(family, type)
            self._socket.connect(filename)
            self._fdin = self._fdout = self._socket.fileno()
        except OSError as error:
            raise OSError("Cannot open image display")

    def close(self):

        """Close image display connection"""

        self._socket.close()


class InetImageDisplay(UnixImageDisplay):

    """INET socket version of image display"""

    def __init__(self, port, hostname=None):
        hostname = hostname or "localhost"
        UnixImageDisplay.__init__(self, (hostname, port), family=socket.AF_INET)


class ImageDisplayProxy(ImageDisplay):

    """Interface to IRAF-compatible image display

    This is a proxy to the actual display that allows retries
    on failures and can switch between display connections.
    """

    def __init__(self, imtdev=None):
        if imtdev:
            self.open(imtdev)
        else:
            self._display = None

    def open(self, imtdev=None):

        """Open image display connection, closing any active connection"""
        self.close()
        self._display = _open(imtdev)

    def close(self):

        """Close active image display connection"""

        if self._display:
            self._display.close()
            self._display = None


    def readCursor(self,sample=0):

        """Read image cursor value for the active image display

        Return immediately if sample is true, or wait for keystroke
        if sample is false (default).  Returns a string with
        x, y, frame, and key.  Opens image display if necessary.
        """

        if not self._display:
            self.open()
        try:
            value = self._display.readCursor(sample)
            # Null value indicates display was probably closed
            if value:
                return value
        except OSError as error:
                pass
        # This error can occur if image display was closed.
        # If a new display has been started then closing and
        # reopening the connection will fix it.  If that
        # fails then give up.
        self.open()
        return self._display.readCursor(sample)

    def setCursor(self,x,y,wcs):
        if not self._display:
            self.open()

        self._display.setCursor(x,y,wcs)

    def checkDisplay(self):
        """ Returns True if a valid connection to a display device is found,
        False if no connection could be found.
        """
        try:
            wcs = self._display.readInfo()
        except:
            return False
        return True



# Print help information
def help():
    print(__doc__)


_display = ImageDisplayProxy()

# create aliases for _display methods

readCursor = _display.readCursor
open = _display.open
close = _display.close
setCursor = _display.setCursor
checkDisplay = _display.checkDisplay
