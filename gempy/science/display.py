# This module contains user level functions related to the displaying
# the input dataset

import os
import sys
from copy import deepcopy
import numpy as np
import numdisplay as nd
from astrodata import AstroData
from astrodata import Errors
from astrodata import Lookups
from astrodata.adutils import gemLog
from astrodata.adutils.gemutil import pyrafLoader
from gempy import geminiTools as gt
from gempy.geminiCLParDicts import CLDefaultParamsDict
from gempy import string as gstr
from gempy.science import resample as rs

# Load the timestamp keyword dictionary that will be used to define the keyword
# to be used for the time stamp for the user level function
timestamp_keys = Lookups.get_lookup_table("Gemini/timestamp_keywords",
                                          "timestamp_keys")

def display_gmos(adinput=None, frame=1, saturation=None, overlay=None):
    """
    This function does a quick tiling if necessary, and calls numdisplay
    to display the data to ds9.
    """

    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()

    # The validate_input function ensures that adinput is not None and returns
    # a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)

    # Initialize the list of output AstroData objects
    adoutput_list = adinput
    
    try:
        if frame is None:
            frame = 1

        # Initialize the local version of numdisplay
        # (overrides the display function to allow for quick overlays)
        lnd = _localNumDisplay()

        for ad in adinput:

            # Check for more than one science extension and tile if found
            nsciext = ad.count_exts("SCI")
            if nsciext!=1:
                disp_ad = deepcopy(ad)
                disp_ad = rs.tile_arrays(adinput=disp_ad, tile_all=True)[0]
            else:
                disp_ad = ad

            sciext = disp_ad["SCI",1]
            data = sciext.data

            masks = []

            # Make saturation mask if desired
            if saturation is not None:
                if saturation=="auto":
                    saturation = disp_ad.saturation_level()
            
                # Check units of 1st science extension; if electrons, 
                # convert saturation limit from ADU to electrons. Also
                # subtract approximate overscan level if needed
                overscan_level = sciext.get_key_value("OVERSCAN")
                if overscan_level is not None:
                    saturation -= overscan_level
                    log.fullinfo("Subtracting overscan level " +
                                 "%.2f from saturation parameter" % 
                                 overscan_level)
                bunit = sciext.get_key_value("BUNIT")
                if bunit=="electron":
                    gain = sciext.gain().as_pytype()
                    saturation *= gain 
                    log.fullinfo("Saturation parameter converted to " +
                                 "%.2f electrons" % saturation)

                # Make saturation mask
                satmask = np.where(data>saturation)
                masks.append(satmask)

            if overlay is not None:
                masks.append(overlay)


            # Display the data
            try:
                lnd.display(data,name=disp_ad.filename,
                            frame=frame,zscale=True,quiet=True,
                            masks=masks, mask_colors=[204,206])
            except:
                log.warning("numdisplay failed")

            frame+=1

        # Return the list of output AstroData objects (unchanged)
        return adoutput_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

   
##############################################################################
# Below are the helper functions for the user level functions in this module #
##############################################################################

class _localNumDisplay(nd.NumDisplay):
    """
    This class overrides the default numdisplay.display function in
    order to implement super-fast overlays.  If this feature can be
    incorporated into numdisplay, this local version should go away.

    mask, if specified, should be a tuple of numpy arrays: the y- and
    x-coordinates of points to be masked.  For example,
    mask = np.where(data>saturation_limit)
    """
    def display(self, pix, name=None, bufname=None, z1=None, z2=None,
                transform=None, zscale=False, contrast=0.25, scale=None,
                masks=None, mask_colors=None,
                offset=None, frame=None, quiet=False):

        """ Displays byte-scaled (UInt8) n to XIMTOOL device.
            This method uses the IIS protocol for displaying the data
            to the image display device, which requires the data to be
            byte-scaled.
            If input is not byte-scaled, it will perform scaling using
            set values/defaults.
        """

        #Ensure that the input array 'pix' is a numpy array
        pix = np.array(pix)
        self.z1 = z1
        self.z2 = z2

        # If any of the display parameters are specified here, apply them
        # if z1 or z2 or transform or scale or offset or frame:
        # If zscale=True (like IRAF's display) selected, calculate z1 and z2 from
        # the data, and clear any transform specified in the call
        # Offset and scale are applied to the data and z1,z2, 
        # so they have no effect on the display
        if zscale:
            if transform != None:
                if not quiet:
                    print "transform disallowed when zscale=True"
                transform = None

            z1, z2 = nd.zscale.zscale(pix, contrast=contrast)

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
            self.z1 = np.minimum.reduce(np.ravel(pix))
        if self.z2 == None:
            self.z2 = np.maximum.reduce(np.ravel(pix))

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
                print 'Error encountered during transformation. No transformation applied...'
            bpix = pix
            self.z1 = np.minimum.reduce(np.ravel(bpix))
            self.z2 = np.maximum.reduce(np.ravel(bpix))
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

        _wcsinfo = nd.displaydev.ImageWCS(bpix,z1=self.z1,z2=self.z2,name=name)
        if not quiet:
            print 'Image displayed with Z1: ',self.z1,' Z2:',self.z2

        bpix = self._fbclipImage(bpix,_d.fbwidth,_d.fbheight)

        # Change pixel value to specified color if desired
        if masks is not None:
            if not isinstance(masks,list):
                masks = [masks]
            if mask_colors is None:
                # Set to red as default
                mask_colors = [204]*len(masks)
            for i in range(len(masks)):
                bpix[masks[i]] = mask_colors[i]

        # Update the WCS to match the frame buffer being used.
        _d.syncWCS(_wcsinfo)

        
        # write out WCS to frame buffer, then erase buffer
        _d.writeWCS(_wcsinfo)

        # Now, send the trimmed image (section) to the display device
        _d.writeImage(bpix,_wcsinfo)

        #displaydev.close()
