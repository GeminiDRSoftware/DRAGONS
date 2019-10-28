#
#                                                                  gemini_python
#
#                                                     primtives_gmos_longslit.py
# ------------------------------------------------------------------------------
import numpy as np
from gempy.gemini import gemini_tools as gt
from gempy.library import astrotools as at

from geminidr.gemini.lookups import DQ_definitions as DQ

from .primitives_gmos_spect import GMOSSpect
from .primitives_gmos_nodandshuffle import GMOSNodAndShuffle
from . import parameters_gmos_longslit

from astropy.modeling import models, fitting
from astropy.stats import sigma_clip

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class GMOSLongslit(GMOSSpect, GMOSNodAndShuffle):
    """
    This is the class containing all of the preprocessing primitives
    for the GMOSLongslit level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = set(["GEMINI", "GMOS", "SPECT", "LS"])

    def __init__(self, adinputs, **kwargs):
        super(GMOSLongslit, self).__init__(adinputs, **kwargs)
        self._param_update(parameters_gmos_longslit)

    def addIllumMaskToDQ(self, adinputs=None, suffix=None, illum_mask=None):
        """
        Adds an illumination mask to each AD object

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        illum_mask: str/None
            name of illumination mask mask (None -> use default)
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        for ad, illum in zip(*gt.make_lists(adinputs, illum_mask, force_ad=True)):
            if ad.phu.get(timestamp_key):
                log.warning('No changes will be made to {}, since it has '
                    'already been processed by addIllumMaskToDQ'.
                            format(ad.filename))
                continue

            if illum is None:
                # Default operation for GMOS LS
                # The 95% cut should ensure that we're sampling something
                # bright (even for an arc)
                # The 75% cut is intended to handle R150 data, where many of
                # the extensions are unilluminated
                row_medians = np.percentile(np.array([np.percentile(ext.data, 95, axis=1)
                                                      for ext in ad]), 75, axis=0)
                rows = np.arange(len(row_medians))
                m_init = models.Polynomial1D(degree=2)
                fit_it = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
                                                           outlier_func=sigma_clip)
                m_final, mask = fit_it(m_init, rows, row_medians)
                mask &= (row_medians < m_final(rows))
                # The default selection tends to mask the edges of the good
                # regions, so rein it in a bit
                mask = at.boxcar(mask, operation=np.logical_and, size=1)
                for ext in ad:
                    ext.mask |= (mask * DQ.unilluminated).astype(DQ.datatype)[:, np.newaxis]

            else:
                log.fullinfo("Using {} as illumination mask".format(illum.filename))
                final_illum = gt.clip_auxiliary_data(ad, aux=illum, aux_type='bpm',
                                          return_dtype=DQ.datatype)

                for ext, illum_ext in zip(ad, final_illum):
                    if illum_ext is not None:
                        # Ensure we're only adding the unilluminated bit
                        iext = np.where(illum_ext.data > 0, DQ.unilluminated,
                                        0).astype(DQ.datatype)
                        ext.mask = iext if ext.mask is None else ext.mask | iext

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)

        return adinputs

