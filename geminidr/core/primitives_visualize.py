import astrodata
import gemini_instruments
from gempy.gemini import gemini_tools as gt
from geminidr.gemini.lookups import DQ_definitions as DQ
from gempy.gemini import eti
try:
    from stsci import numdisplay as nd
except ImportError:
    import numdisplay as nd

import numpy as np

from geminidr import PrimitivesBASE
from .parameters_visualize import ParametersVisualize

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class Visualize(PrimitivesBASE):
    """
    This is the class containing the visualization primitives.
    """
    tagset = set(["GEMINI"])

    def __init__(self, adinputs, context, ucals=None, uparms=None):
        super(Visualize, self).__init__(adinputs, context, ucals=ucals,
                                         uparms=uparms)
        self.parameters = ParametersVisualize

    def display(self, adinputs=None, stream='main', **params):
        """
        Displays an image

        Parameters
        ----------
        extname: str
        frame: int
        ignore: bool
            setting to True turns off the display
        remove_bias: bool
        threshold: str
        tile: bool
        zscale: bool
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        pars = self.parameters.display

        # No-op if ignore=True
        if pars["ignore"]:
            log.warning("display turned off per user request")
            return

        threshold = pars['threshold']
        remove_bias = pars['remove_bias']
        extname = pars['extname']
        tile = pars['tile']
        zscale = pars['zscale']

        # We may be manipulating the data significantly, so the best option
        # is to create a new PrimitivesClass instance and work with that
        p = self.__class__(self.adinputs, self.context)

        # Threshold and bias make sense only for SCI extension
        if extname != 'SCI':
            threshold = None
            remove_bias = False
        elif threshold == 'None':
            threshold = None
        elif threshold == 'auto':
            mosaicked = [(ad.phu.get(self.timestamp_keys["mosaicDetectors"])
                          is not None) or
                         (ad.phu.get(self.timestamp_keys["tileArrays"])
                          is not None) for ad in p.adinputs]
            has_dq = [all(ext.mask is not None for ext in ad)
                      for ad in p.adinputs]
            if not all(has_dq):
                if any([m and not d] for m,d in zip(mosaicked, has_dq)):
                    log.warning("Cannot add DQ to mosaicked data; no "
                                "threshold mask will be applied")
                    threshold = None
                else:
                    # At least one input lacks DQ; these inputs have not been
                    # mosaicked. addDQ will no-op any previously dqAdded input
                    p.addDQ()

        if remove_bias:
            for ad in p.adinputs:
                if (ad.phu.get('BIASIM') or ad.phu.get('DARKIM') or
                    any(ad.hdr.get('OVERSCAN'))):
                    log.fullinfo("Bias level has already been removed from "
                                 "data; no approximate correction will be "
                                 "performed")
                else:
                    try:
                        bias_level = gemini_instruments.gmos.pixel_functions.get_bias_level(ad)
                    except NotImplementedError:
                        # For non-GMOS instruments
                        bias_level = None

                    if bias_level is not None:
                        log.stdinfo("Subtracting approximate bias level from "
                                    "{} for display".format(ad.filename))
                        log.fullinfo("Bias levels used: {}".str(bias_level))
                        for ext, bias in zip(ad, bias_level):
                            ext.add(bias)
                    else:
                        log.warning("Bias level not found for {}; approximate "
                                    "bias will not be removed".format(ad.filename))

        # Check whether data needs to be tiled before displaying
        # Otherwise, flatten all desired extensions into a single list
        if tile:
            if any(len(ad)>1 for ad in p.adinputs):
                log.fullinfo("Tiling extensions together before displaying")
                p.tileArrays()
        else:
            p.adinputs = [ext for ad in p.adinputs for ext in ad]

        # Get overlays from RC if available (eg. IQ overlays made by measureIQ)
        # then clear them out so they don't persist to the next display call
        # TODO: May need to come back to this
        overlay_lists = gt.make_lists(p.adinputs, getattr(self, 'overlay', None))
        self.overlay = None

        frame = pars['frame'] if pars['frame'] else 1
        lnd = _localNumDisplay()

        for ad, overlay in zip(*overlay_lists):
            if frame > 16:
                log.warning("Too many images; only the first 16 are displayed")
                break

            if len(ad) > 1:
                raise IOError("Found {} extensions for {}[{}]; exactly 1 is "
                              "required".format(len(ad), ad.filename, extname))

            # Squeeze the data to remove any empty dimensions (eg, raw F2 data)
            ad.operate(np.squeeze)

            # Get the data we're going to display. TODO Replace extname with attr?
            data = getattr(ad, {'SCI':'data', 'DQ':'mask',
                                'VAR':'variance'}[extname], None)
            if data is None:
                log.warning("No data to display in {}[{}]".format(ad.filename,
                                                                  extname))
                continue

            # One-dimensional data (ie, extracted spectra)
            if len(data.shape) == 1:
                continue

            # Make threshold mask if desired
            masks = []
            mask_colors = []
            if threshold is not None:
                if threshold != 'auto':
                    satmask = data > float(threshold)
                else:
                    dqdata = ad.mask
                    if dqdata is None:
                        log.warning("No DQ plane found; cannot make "
                                    "threshold mask")
                        satmask = None
                    else:
                        satmask = (dqdata & (DQ.non_linear | DQ.saturated)) > 0
                if satmask is not None:
                    masks.append(satmask)
                    mask_colors.append(204)

            if overlay:
                masks.append(overlay)
                mask_colors.append(206)

            # Define the display name
            if tile and extname=='SCI':
                name = ad.filename
            elif tile:
                name = '{}({})'.format(ad.filename, extname)
            else:
                name = '{}({},{})'.format(ad.filename, extname, ad.hdr.EXTVER)

            try:
                lnd.display(data, name=name, frame=frame, zscale=zscale,
                            quiet=True, masks=masks, mask_colors=mask_colors)
            except IOError:
                log.warning("ds9 not found; cannot display input")

            frame += 1

            # Print from statistics for flats
            if extname=='SCI' and {'GMOS', 'IMAGE', 'FLAT'}.issubset(ad.tags):
                good_data = np.ma.masked_array(ad.data, mask=ad.mask)
                mean = np.ma.mean(good_data)
                median = np.ma.median(good_data)
                # Bug in numpy v1.9 where ma.median returns an array
                if isinstance(median, np.ndarray):
                    median = median[0]
                log.stdinfo("Twilight flat counts for {}:".format(ad.filename))
                log.stdinfo("    Mean value:   {:.0f}".format(mean))
                log.stdinfo("    Median value: {:.0f}".format(median))
        return

    def mosaicDetectors(self, adinputs=None, stream='main', **params):
        pass


    def tileArrays(self, adinputs=None, stream='main', **params):
        pass

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
    TODO: Can it be an array of booleans, the same size as the data?
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