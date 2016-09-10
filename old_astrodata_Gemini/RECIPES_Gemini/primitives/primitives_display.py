from copy import deepcopy
import numpy as np
try:
    from stsci import numdisplay as nd
except ImportError:
    import numdisplay as nd

from astrodata.utils import Errors
from astrodata.utils import Lookups
from astrodata.utils import logutils
from astrodata.utils.gemutil import pyrafLoader

from gempy.gemini import gemini_data_calculations as gdc
from gempy.gemini import gemini_tools as gt
from gempy.gemini import eti

import pifgemini.standardize as sdz
import pifgemini.gmos as gm

from primitives_GENERAL import GENERALPrimitives

class DisplayPrimitives(GENERALPrimitives):
    """
    This is the class containing all of the display primitives
    for the GEMINI level of the type hierarchy tree. It inherits
    all the primitives from the level above, 'GENERALPrimitives'.
    """
    astrotype = "GEMINI"
    
    def init(self, rc):
        GENERALPrimitives.init(self, rc)
        return rc
    init.pt_hide = True
    
    def display(self, rc):
        # Instantiate the log
        log = logutils.get_logger(__name__)

        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "display", "starting"))
        
        # Get parameters from RC
        ignore = rc["ignore"]
        threshold = rc["threshold"]
        remove_bias = rc["remove_bias"]

        # Get inputs
        adinput = rc.get_inputs_as_astrodata()
        orig_input = adinput
        deepcopied = False
        
        # This allows one to control whether the images are displayed
        # or not without having to change the recipe.  To turn off
        # the display primitive in a recipe from reduce: 
        # -p display:ignore=True
        if ignore:
            log.warning("display turned off per user request")
            rc.report_output(orig_input)
            yield rc
            return

        # Threshold and bias parameters only make sense for SCI extension;
        # turn it off for others
        extname = rc["extname"]
        if extname!="SCI":
            threshold = None
            remove_bias = False
        elif threshold=="None":
            threshold = None
        elif threshold=="auto":
            dqext = np.array([ad["DQ"] for ad in adinput])
            mosaic = np.array([((ad.phu_get_key_value(
                                        self.timestamp_keys["mosaicDetectors"])
                                 is not None)
                                or
                                (ad.phu_get_key_value(
                                        self.timestamp_keys["tileArrays"])
                                 is not None))
                               for ad in adinput])
            if not np.all(dqext):
                if not np.any(mosaic):
                    # This is the first possible modification to the data;
                    # always deepcopy before proceeding
                    adinput = [deepcopy(ad) for ad in orig_input]
                    deepcopied = True

                    adinput = sdz.add_dq(adinput, bpm=None,
                                         copy_input=False, index=rc["index"])
                    if not isinstance(adinput,list):
                        adinput = [adinput]
                else:
                    log.warning("Cannot add DQ plane to mosaicked data; " \
                                "no threshold mask will be displayed")
                    threshold=None

        # Check whether approximate bias level should be removed
        if remove_bias:
            # Copy the original input if necessary, before
            # modifying it
            if not deepcopied:
                adinput = [deepcopy(ad) for ad in orig_input]
                deepcopied = True

            new_adinput = []
            for ad in adinput:
                # Check whether data has been bias- or dark-subtracted
                biasim = ad.phu_get_key_value("BIASIM")
                darkim = ad.phu_get_key_value("DARKIM")

                # Check whether data has been overscan-subtracted
                overscan = np.array([ext.get_key_value("OVERSCAN") 
                                     for ext in ad["SCI"]])
                if np.any(overscan) or biasim or darkim:
                    log.fullinfo("Bias level has already been removed "
                                 "from data; no approximate correction "
                                 "will be performed")
                else:
                    # Get the bias level
                    try:
                        bias_level = gdc.get_bias_level(adinput=ad)
                    except NotImplementedError:
                        log.fullinfo(sys.exc_info()[1])
                        bias_level = None

                    if bias_level is not None:
                        # Subtract the bias level from each science extension
                        log.stdinfo("Subtracting approximate bias level "
                                     "from %s for display" % ad.filename)
                        log.fullinfo("Bias levels used: %s" % str(bias_level))
                        ad = ad.sub(bias_level)
                    else:
                        log.warning("Bias level not found for %s; "
                                    "approximate bias will not be removed" % 
                                    ad.filename)

                new_adinput.append(ad)
            adinput = new_adinput

        # Check whether data needs to be tiled before displaying
        # Otherwise, flatten all desired extensions into a single list
        tile = rc["tile"]
        if tile:
            next = np.array([ad.count_exts(extname) for ad in adinput])
            if np.any(next>1):
                log.fullinfo("Tiling extensions together before displaying")
                if not deepcopied:
                    adinput = [deepcopy(ad) for ad in orig_input]
                    deepcopied = True

                adinput = gm.tile_arrays(adinput, tile_all=True,
                                         copy_input=False, index=rc["index"])
                if not isinstance(adinput,list):
                    adinput = [adinput]
        else:
            extinput = []
            for ad in adinput:
                exts = ad[extname]
                if exts is None:
                    continue
                for ext in exts:
                    if extname=="SCI" and threshold=="auto":
                        dqext = ad["DQ",ext.extver()]
                        if dqext is not None:
                            ext.append(dqext)
                    extinput.append(ext)
            adinput = extinput

        # Get overlays from RC if available (eg. IQ overlays made by measureIQ)
        # then clear them out so they don't persist to the next display call
        overlay_dict = gt.make_dict(key_list=adinput, value_list=rc["overlay"])
        rc["overlay"] = None

        # Set the starting frame
        frame = rc["frame"]
        if frame is None:
            frame = 1

        # Initialize the local version of numdisplay
        # (overrides the display function to allow for quick overlays)
        lnd = _localNumDisplay()

        # Loop over each input AstroData object in the input list
        if len(adinput)<1:
            log.warning("No extensions to display with extname %s" % extname)
        for ad in adinput:
              
            if frame>16:
                log.warning("Too many images; only the first 16 are displayed.")
                break

            # Check for more than one extension
            ndispext = ad.count_exts(extname)
            if ndispext==0:
                log.warning("No extensions to display in "\
                                "%s with extname %s" %
                            (ad.filename,extname))
                continue
            elif ndispext>1:
                raise Errors.InputError("Found %i extensions for "\
                                        "%s[%s]; exactly 1 is required" %
                                        (ndispext,ad.filename,extname))

            dispext = ad[extname]
            
            # Squeeze the data to get rid of any empty dimensions
            # (eg. in raw F2 data)
            data = np.squeeze(dispext.data)

            # Check for 1-D data (ie. extracted spectra)
            if len(data.shape)==1:
                
                # Use splot to display instead of numdisplay
                log.fullinfo("Calling IRAF task splot to display data")
                splot_task = eti.sploteti.SplotETI(rc,ad)
                splot_task.run()
                continue

            # Make threshold mask if desired
            masks = []
            mask_colors = []
            if threshold is not None:
            
                if threshold!="auto":
                    # Make threshold mask from user supplied value;
                    # Assume units match units of data
                    threshold = float(threshold)
                    satmask = np.where(data>threshold)
                else:
                    # Make the mask from the nonlinear and 
                    # saturation bits in the DQ plane
                    dqext = ad["DQ",dispext.extver()]
                    if dqext is None:
                        log.warning("No DQ plane found; cannot make threshold "\
                                    "mask")
                        satmask = None
                    else:
                        dqdata = np.squeeze(dqext.data)
                        satmask = np.where(np.logical_or(dqdata & 2,
                                                         dqdata & 4))
                if satmask is not None:
                    masks.append(satmask)
                    mask_colors.append(204)

            overlay = overlay_dict[ad]
            if overlay is not None:
                masks.append(overlay)
                mask_colors.append(206)

                # ds9 color codes: should make this into a dictionary
                # and allow user to specify
                #
                # red = 204
                # green = 205
                # blue = 206
                # yellow = 207
                # light blue = 208
                # magenta = 209
                # orange = 210
                # dark pink = 211
                # bright orange = 212
                # light yellow = 213
                # pink = 214
                # blue-green = 215
                # pink = 216
                # peach = 217


            # Define the display name
            if tile and extname=="SCI":
                name = ad.filename
            elif tile:
                # numdisplay/ds9 doesn't seem to like square brackets
                # or spaces in the name, so use parentheses for extension
                name = "%s(%s)" % (ad.filename,extname)
            else:
                name = "%s(%s,%d)" % (ad.filename,extname,dispext.extver())

            # Display the data
            try:
                lnd.display(data,name=name,
                            frame=frame,zscale=rc["zscale"],quiet=True,
                            masks=masks, mask_colors=mask_colors)
            except IOError:
                log.warning("DS9 not found; cannot display input")

            frame+=1
        
            # Print some statistics for flats
            if "GMOS_IMAGE_FLAT" in ad.types and extname=="SCI":
                scidata = ad["SCI"].data
                dqext = ad["DQ"]
                if dqext is not None:
                    dqdata = dqext.data
                    good_data = scidata[dqdata==0]
                else:
                    good_data = scidata

                log.stdinfo("Twilight flat counts for %s:" % ad.filename)
                log.stdinfo("    Mean value:   %.0f" % np.mean(good_data, dtype=np.float64))
                log.stdinfo("    Median value: %.0f" % np.median(good_data))

        rc.report_output(orig_input)
        yield rc

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
    mask = np.where(data>threshold)
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
                if (masks[i][0].size>0 and masks[i][1].size>0):
                    bpix[masks[i]] = mask_colors[i]

        # Update the WCS to match the frame buffer being used.
        _d.syncWCS(_wcsinfo)

        
        # write out WCS to frame buffer, then erase buffer
        _d.writeWCS(_wcsinfo)

        # Now, send the trimmed image (section) to the display device
        _d.writeImage(bpix,_wcsinfo)

        #displaydev.close()
