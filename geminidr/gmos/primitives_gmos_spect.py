#
#                                                                  gemini_python
#
#                                                        primtives_gmos_spect.py
# ------------------------------------------------------------------------------
import numpy as np

from geminidr.gemini.lookups import DQ_definitions as DQ
from gempy.gemini import gemini_tools as gt

from geminidr.core import Spect
from .primitives_gmos import GMOS
from . import parameters_gmos_spect

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class GMOSSpect(GMOS, Spect):
    """
    This is the class containing all of the preprocessing primitives
    for the GMOSImage level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = set(["GEMINI", "GMOS", "SPECT"])

    def __init__(self, adinputs, **kwargs):
        super(GMOSSpect, self).__init__(adinputs, **kwargs)
        self._param_update(parameters_gmos_spect)

    def findAcquisitionSlits(self, adinputs=None, **params):
        """
        This primitive determines which rows of a 2D spectroscopic frame
        contain the stars used for target acquisition, primarily so they can
        be used later to estimate the image FWHM. This is done by cross-
        correlating a vertical cut of the image with a cartoon model of the
        slit locations determined from the MDF.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        for ad in adinputs:
            # First, check we want to process this: not if it's already been
            # processed; or has no MDF; or has no acquisition stars in the MDF
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by findAcqusitionSlits".
                            format(ad.filename))
                continue

            try:
                mdf = ad.MDF
            except AttributeError:
                log.warning("No MDF associated with {}".format(ad.filename))
                continue

            if 'priority' not in mdf.columns:
                log.warning("No acquisition slits in {}".format(ad.filename))
                continue

            # Tile and collapse along wavelength direction
            ad_tiled = self.tileArrays([ad], tile_all=True)[0]

            # Ignore bad pixels (non-linear/saturated are OK)
            if ad_tiled[0].mask is None:
                mask = None
            else:
                mask = ad_tiled[0].mask & ~(DQ.non_linear | DQ.saturated)
            spatial_profile = np.ma.array(ad_tiled[0].data,
                                          mask=mask).sum(axis=1)

            # Construct a theoretical illumination map from the MDF data
            slits_profile = np.zeros_like(spatial_profile)
            image_pix_scale = ad.pixel_scale()

            shuffle = ad.nod_pixels() // ad.detector_y_bin()
            # It is possible to use simply the MDF information in mm to get
            # the necessary slit position data, but this relies on knowing
            # the distortion correction. It seems better to use the MDF
            # pixel information, if it exists.
            try:
                mdf_pix_scale = mdf.meta['header']['PIXSCALE']
            except KeyError:
                mdf_pix_scale = ad.pixel_scale() / ad.detector_y_bin()
            # There was a problem with the mdf_pix_scale for GMOS-S pre-2009B
            # Work around this because the two pixel scales should be in a
            # simple ratio (3:2, 2:1, etc.)
            ratios = np.array([1.*a/b for a in range(1,6) for b in range(1,6)])
            # Here we have to account for the EEV->Hamamatsu change
            # (I've future-proofed this for the same event on GMOS-N)
            ratios = np.append(ratios,[ratios*0.73/0.8,ratios*0.727/0.807])
            nearest_ratio = ratios[np.argmin(abs(mdf_pix_scale /
                                                 image_pix_scale - ratios))]
            # -1 because python is zero-indexed (see +1 later)
            slits_y = mdf['y_ccd'] * nearest_ratio - 1

            try:
                    slits_width = mdf['slitsize_y']
            except KeyError:
                    slits_width = mdf['slitsize_my'] * 1.611444

            for (slit, width) in zip(slits_y, slits_width):
                slit_ymin = slit - 0.5*width/image_pix_scale
                slit_ymax = slit + 0.5*width/image_pix_scale
                # Only add slit if it wasn't shuffled off top of CCD
                if slit < ad_tiled[0].data.shape[0]-shuffle:
                    slits_profile[max(int(slit_ymin),0):
                                  min(int(slit_ymax+1),len(slits_profile))] = 1
                    if slit_ymin > shuffle:
                        slits_profile[int(slit_ymin-shuffle):
                                      int(slit_ymax-shuffle+1)] = 1

            # Cross-correlate collapsed image with theoretical profile
            c = np.correlate(spatial_profile, slits_profile, mode='full')
            slit_offset = np.argmax(c)-len(spatial_profile) + 1

            # Work out where the alignment slits actually are!
            # NODAYOFF should possibly be incorporated here, to better estimate
            # the locations of the positive traces, but I see inconsistencies
            # in the sign (direction of +ve values) for different datasets.
            acq_slits = np.logical_and(mdf['priority']=='0',
                                       slits_y<ad_tiled[0].data.shape[0]-shuffle)
            # Slits centers and widths
            acq_slits_y = (slits_y[acq_slits] + slit_offset + 0.5).astype(int)
            acq_slits_width = (slits_width[acq_slits] / image_pix_scale +
                               0.5).astype(int)
            star_list = ' '.join('{}:{}'.format(y,w) for y,w in
                                 zip(acq_slits_y,acq_slits_width))

            ad.phu.set('ACQSLITS', star_list,
                       comment=self.keyword_comments['ACQSLITS'])

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)
        return adinputs