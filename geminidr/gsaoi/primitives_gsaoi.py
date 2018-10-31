#
#                                                                  gemini_python
#
#                                                      primitives_gsaoi_image.py
# ------------------------------------------------------------------------------
import numpy as np
from os import path
from astropy.table import vstack

from gempy.gemini import gemini_tools as gt

from ..core import NearIR
from geminidr.gemini.primitives_gemini import Gemini
from . import parameters_gsaoi

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class GSAOI(Gemini, NearIR):
    """
    This is the class containing all of the preprocessing primitives
    for the F2 level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = set(["GEMINI", "GSAOI"])

    def __init__(self, adinputs, **kwargs):
        super(GSAOI, self).__init__(adinputs, **kwargs)
        self.inst_lookups = 'geminidr.gsaoi.lookups'
        self._param_update(parameters_gsaoi)

    def standardizeInstrumentHeaders(self, adinputs=None, suffix=None):
        """
        This primitive is used to make the changes and additions to the
        keywords in the headers of GSAOI data, specifically.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by "
                            "standardizeInstrumentHeaders".format(ad.filename))
                continue

            # Standardize the headers of the input AstroData object. Update the
            # keywords in the headers that are specific to GSAOI.
            log.status("Updating keywords that are specific to GSAOI")

            # Filter name (required for IRAF?)
            ad.phu.set('FILTER', ad.filter_name(stripID=True, pretty=True),
                       self.keyword_comments['FILTER'])

            # Pixel scale
            ad.phu.set('PIXSCALE', ad.pixel_scale(), self.keyword_comments['PIXSCALE'])

            for desc in ('read_noise', 'gain', 'non_linear_level',
                         'saturation_level'):
                kw = ad._keyword_for(desc)
                for ext, value in zip(ad, getattr(ad, desc)()):
                    ext.hdr.set(kw, value, self.keyword_comments[kw])

            # Move BUNIT from PHU to the extension HDUs
            try:
                bunit = ad.phu['BUNIT']
            except KeyError:
                pass
            else:
                del ad.phu['BUNIT']
                ad.hdr.set('BUNIT', bunit, self.keyword_comments['BUNIT'])

            # There is a bug in GSAOI data where the central 1K x 1K ROI is
            # read out (usually for photometric standards) and the CCDSEC
            # keyword needs to be fixed for this type of data so the auxiliary
            # data (i.e. BPM and flats) match. Best guess date for start of
            # problem is currently May 15 2013. It is not clear if this bug is
            # in the detector controller code or the SDSU board timing.
            if ad.phu.get('DATE-OBS') >= '2013-05-13':
                for ext in ad:
                    if ext.array_section(pretty=True) == '[513:1536,513:1536]':
                        log.stdinfo("Updating the CCDSEC for central ROI data")
                        y1o = 513 if ext.hdr['EXTVER'] < 3 else 511
                        y2o = y1o + 1024
                        secstr = "[{0}:{1},{2}:{3}]".format(513, 1536, y1o+1, y2o)
                        ext.hdr.set('CCDSEC', secstr, self.keyword_comments['CCDSEC'])

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

    # def tileArrays(self, adinputs=None, **params):
    #     """
    #     This primitive tiles the four GSAOI arrays, producing a single
    #     extension. The tiling is very approximate, and primarily for display
    #     purposes. Any attached OBJCAT has its pixel coordinates updated.
    #
    #     Parameters
    #     ----------
    #     suffix: str
    #         suffix to be added to output files
    #     tile_all: bool
    #         not relevant here (False no-ops)
    #     """
    #     log = self.log
    #     log.debug(gt.log_message("primitive", self.myself(), "starting"))
    #     timestamp_key = self.timestamp_keys[self.myself()]
    #     tile_all = params["tile_all"]
    #
    #     for ad in adinputs:
    #         if not tile_all or len(ad) == 1:
    #             log.fullinfo("Only one science extension found or tile_all "
    #                     "disabled;\n no tiling done for {}".format(ad.filename))
    #             continue
    #
    #         # First trim off any unused border regions still present
    #         # so they won't get tiled with science data:
    #         log.fullinfo("Trimming to data section:")
    #         ad = gt.trim_to_data_section(ad, keyword_comments=self.keyword_comments)
    #
    #         # Determine output size and info to determine locations of arrays
    #         # Gap size estimated at 3.0 arcsec from Disco-Stu output tiles
    #         detsec_list = ad.detector_section()
    #         gap_size = int(3.0 / ad.pixel_scale())
    #         x1 = min(s.x1 for s in detsec_list)
    #         x2 = max(s.x2 for s in detsec_list)
    #         y1 = min(s.y1 for s in detsec_list)
    #         y2 = max(s.y2 for s in detsec_list)
    #         output_shape = (y2-y1+gap_size, x2-x1+gap_size)
    #
    #         shifts = [(d.x1-x1 if d.x1<1024 else d.x1-x1+gap_size,
    #                    d.y1-y1 if d.y1<1024 else d.y1-y1+gap_size)
    #                    for d in detsec_list]
    #
    #         # Annoyingly, it's not (yet) possible to assign data to a section
    #         # of an NDData object, so have to treat data, mask, and variance
    #         # as arrays separately
    #         adout = {'data': None, 'mask': None, 'variance': None, 'OBJMASK': None}
    #         for attr in adout:
    #             if all(getattr(ext, attr, None) is not None for ext in ad):
    #                 tiled = np.full(output_shape, 16, dtype=np.int16) if \
    #                     attr=='mask' else np.zeros(output_shape, dtype=np.float32)
    #                 for ext, shift in zip(ad, shifts):
    #                     ox1, oy1 = shift
    #                     ox2 = ox1 + ext.data.shape[1]
    #                     oy2 = oy1 + ext.data.shape[0]
    #                     tiled[oy1:oy2, ox1:ox2] = getattr(ext, attr)
    #                 adout.update({attr: tiled})
    #         tiled_objcat = _tile_objcat(ad, shifts)
    #
    #         # Reset the AD object. Do it in place... why not?
    #         ad[0].reset(data=adout['data'], mask=adout['mask'],
    #                     variance=adout['variance'])
    #         if adout['OBJMASK'] is not None:
    #             ad[0].OBJMASK = adout['OBJMASK']
    #         # OBJCAT is left undefined if no input extension had one
    #         if hasattr(ad[0], 'OBJCAT') or tiled_objcat:
    #             ad[0].OBJCAT = tiled_objcat
    #         for index in range(1, len(ad)):
    #             del ad[1]
    #
    #         # These are no longer valid
    #         del ad.hdr['CCDNAME']
    #         try:
    #             del ad.hdr['TRIMSEC']
    #         except (KeyError, AttributeError):
    #             pass
    #
    #         # Update geometry keywords
    #         for kw in ('DATASEC', 'CCDSEC'):
    #             ad.hdr.set(kw, '[1:{1},1:{0}]'.format(*ad[0].data.shape),
    #                        comment=self.keyword_comments[kw])
    #
    #         # This doesn't match the array dimensions, due to the gaps, but
    #         # it represents the range of the full, contiguous detector
    #         # mosaic that is spanned by the tiled data.
    #         ad.hdr.set('DETSEC', '[{}:{},{}:{}]'.format(x1+1,x2, y1+2,y2),
    #                    comment=self.keyword_comments['DETSEC'])
    #
    #         # Update the CRPIXn keywords
    #         for n, off in enumerate(shifts[0], start=1):
    #             key = 'CRPIX{}'.format(n)
    #             crpix = ad[0].hdr.get(key)
    #             if crpix is not None:
    #                 ad.hdr.set(key, crpix+off, self.keyword_comments[key])
    #
    #         # Timestamp and update header
    #         gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
    #         ad.update_filename(suffix=params["suffix"], strip=True)
    #
    #     return adinputs

    @staticmethod
    def _has_valid_extensions(ad):
        """Check the AD has a valid number of extensions"""
        return len(ad) == 4

    def makeBPM(self, adinputs=None, **params):
        """
        To be run from recipe makeProcessedBPM.

        The main input is a flat field image that has been constructed by
        stacking the differences of lamp on / off exposures in a given filter
        and normalizing the resulting image to unit average.

        A 'darks' stream must also be provided, containing a single image
        constructed by stacking short darks.


        Parameters
        ----------
        dark_hi_thresh: float, optional
            Maximum data value above which pixels in the input dark are
            considered bad. For GSAOI (with the default value of None), hot
            pixels are considered to be those with a level greater than 75% of
            (the mean + 3 sigma). If the user sets this parameter to a number,
            however, that absolute limit (always in ADUs) will be used instead.
        flat_lo_thresh: float, optional
            Minimum (unit-normalized) data value below which pixels in the
            input flat are considered to be bad (default 0.5).

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        # This GSAOI version approximately follows a procedure documented on
        # the GSAOI Web pages (GSAOI_BPM_forweb.pdf). That document is
        # inconsistent on whether a mean or median is used for the dark, but
        # it seems to make very little difference to the result (and the actual
        # example commands shown appear to use a mean).

        dark_hi = params['dark_hi_thresh']
        flat_lo = params['flat_lo_thresh']

        # Get the stacked flat and dark; these are single-element lists
        try:
            flat = adinputs[0]
        except (KeyError, TypeError):
            raise IOError("A SET OF FLATS IS REQUIRED INPUT")
        try:
            dark = self.streams['darks'][0]
        except (KeyError, TypeError):
            raise IOError("A SET OF DARKS IS REQUIRED INPUT")

        for dark_ext, flat_ext in zip(dark, flat):

            msg = "BPM flat mask lower limit: {}"
            log.stdinfo(msg.format(flat_lo))

            flat_mask = flat_ext.data < flat_lo  # (already normalized)

            msg = "BPM dark mask upper limit: {:.2f} ADU ({:.2f})"

            bunit = dark_ext.hdr.get('BUNIT', 'ADU').upper()
            if bunit in ('ELECTRON', 'ELECTRONS'):
                conv = dark_ext.gain()
            elif bunit == 'ADU' or dark_hi is None:
                conv = 1.
            else:
                raise ValueError("Input units for dark should be ADU or "
                                 "ELECTRON, not {}".format(bunit))

            if dark_hi is None:
                # Use the "standard" calculation for GSAOI:
                dark_lim = 0.75 * (np.mean(dark_ext.data) \
                                      + 3 * np.std(dark_ext.data))
            else:
                # Convert a user-specified threshold from ADUs:
                dark_lim = conv * dark_hi

            log.stdinfo(msg.format(dark_lim / conv, dark_lim))

            dark_mask = dark_ext.data > dark_lim

            # combine masks and write to bpm file
            data_mask = np.ma.mask_or(dark_mask, flat_mask)
            flat_ext.reset(data_mask.astype(np.int16), mask=None, variance=None)

        flat.update_filename(suffix="_bpm", strip=True)
        flat.phu.set('OBJECT', 'BPM')
        gt.mark_history(flat, primname=self.myself(), keyword=timestamp_key)
        return [flat]


##############################################################################
# Below are the helper functions for the primitives in this module           #
##############################################################################

def _tile_objcat(ad, shifts):
    """
    This produces a single Table instance combining all the individual
    OBJCATs, with X_IMAGE and Y_IMAGE updated to account for the tiling,
    and NUMBER changed to avoid duplications.

    Parameters
    ----------
    ad: astrodata
        input AD instance (with OBJCATs)
    shifts: list of 2-tuples
        array shifts (x,y) from original extension into tiled image

    Returns
    -------
    Table: the tiled OBJCAT
    """
    tiled_objcat = None
    for ext, shift in zip(ad, shifts):
        try:
            objcat = ext.OBJCAT
        except AttributeError:
            pass
        else:
            objcat['X_IMAGE'] += shift[0]
            objcat['Y_IMAGE'] += shift[1]
            if tiled_objcat:
                objcat['NUMBER'] += max(tiled_objcat['NUMBER'])
                tiled_objcat = vstack([tiled_objcat, objcat],
                                      metadata_conflicts='silent')
            else:
                tiled_objcat = objcat

    return tiled_objcat
