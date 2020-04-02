#
#                                                                  gemini_python
#
#                                                     primtives_gmos_longslit.py
# ------------------------------------------------------------------------------
import numpy as np
import warnings
import astrodata

from gempy.gemini import gemini_tools as gt
from gempy.library import astrotools as at
from gempy.library import astromodels

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
    tagset = {"GEMINI", "GMOS", "SPECT", "LS"}

    def __init__(self, adinputs, **kwargs):
        super().__init__(adinputs, **kwargs)
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

            if illum:
                log.fullinfo("Using {} as illumination mask".format(illum.filename))
                final_illum = gt.clip_auxiliary_data(ad, aux=illum, aux_type='bpm',
                                          return_dtype=DQ.datatype)

                for ext, illum_ext in zip(ad, final_illum):
                    if illum_ext is not None:
                        # Ensure we're only adding the unilluminated bit
                        iext = np.where(illum_ext.data > 0, DQ.unilluminated,
                                        0).astype(DQ.datatype)
                        ext.mask = iext if ext.mask is None else ext.mask | iext
            elif not all(detsec.y1 > 1600 and detsec.y2 < 2900
                         for detsec in ad.detector_section()):
                # Default operation for GMOS LS
                # The 95% cut should ensure that we're sampling something
                # bright (even for an arc)
                # The 75% cut is intended to handle R150 data, where many of
                # the extensions are unilluminated
                row_medians = np.percentile(np.array([np.percentile(ext.data, 95, axis=1)
                                                      for ext in ad]), 75, axis=0)
                rows = np.arange(len(row_medians))
                m_init = models.Polynomial1D(degree=3)
                fit_it = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
                                                           outlier_func=sigma_clip)
                m_final, _ = fit_it(m_init, rows, row_medians)
                # Find points which are significantly below the smooth illumination fit
                row_mask = at.boxcar(m_final(rows) - row_medians > 0.1 * np.median(row_medians),
                                     operation=np.logical_or, size=2)
                for ext in ad:
                    ext.mask |= (row_mask * DQ.unilluminated).astype(DQ.datatype)[:, np.newaxis]

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)

        return adinputs

    def normalizeFlat(self, adinputs=None, **params):
        """
        This primitive normalizes a GMOS Longslit spectroscopic flatfield
        in a manner similar to that performed by gsflat in Gemini-IRAF.
        A cubic spline is fitted along the dispersion direction of each
        row, separately for each CCD.

        As this primitive is GMOS-specific, we know the dispersion direction
        will be along the rows, and there will be 3 CCDs.

        For Hamamatsu CCDs, the 21 unbinned columns at each CCD edge are
        masked out, following the procedure in gsflat.
        TODO: Should we add these in the BPM?

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        spectral_order: int/str
            order of fit in spectral direction
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        # For flexibility, the code is going to pass whatever validated
        # parameters it gets (apart from suffix and spectral_order) to
        # the spline fitter
        spline_kwargs = params.copy()
        suffix = spline_kwargs.pop("suffix")
        spectral_order = spline_kwargs.pop("spectral_order")
        threshold = spline_kwargs.pop("threshold")

        # Parameter validation should ensure we get an int or a list of 3 ints
        try:
            orders = [int(x) for x in spectral_order]
        except TypeError:
            orders = [spectral_order] * 3

        for ad in adinputs:
            xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()
            array_info = gt.array_information(ad)
            is_hamamatsu = 'Hamamatsu' in ad.detector_name(pretty=True)
            ad_tiled = self.tileArrays([ad], tile_all=False)[0]
            ad_fitted = astrodata.create(ad.phu)
            for ext, order, indices in zip(ad_tiled, orders, array_info.extensions):
                # If the entire row is unilluminated, we want to fit
                # the pixels but still keep the edges masked
                try:
                    ext.mask ^= (np.bitwise_and.reduce(ext.mask, axis=1) & DQ.unilluminated)[:, None]
                except TypeError:  # ext.mask is None
                    pass
                else:
                    if is_hamamatsu:
                        ext.mask[:, :21 // xbin] = 1
                        ext.mask[:, -21 // xbin:] = 1
                fitted_data = np.empty_like(ext.data)
                pixels = np.arange(ext.shape[1])

                for i, row in enumerate(ext.nddata):
                    masked_data = np.ma.masked_array(row.data, mask=row.mask)
                    weights = np.sqrt(np.where(row.variance > 0, 1. / row.variance, 0.))
                    spline = astromodels.UnivariateSplineWithOutlierRemoval(pixels, masked_data,
                                                    order=order, w=weights, **spline_kwargs)
                    fitted_data[i] = spline(pixels)
                # Copy header so we have the _section() descriptors
                # Turn zeros into tiny numbers to avoid 0/0 errors and NaNs
                ad_fitted.append(fitted_data + np.spacing(0), header=ext.hdr)

            # Find the largest spline value for each row across all extensions
            # and mask pixels below the requested fraction of the peak
            row_max = np.array([ext_fitted.data.max(axis=1)
                                for ext_fitted in ad_fitted]).max(axis=0)

            # Prevent runtime error in division
            row_max[row_max == 0] = np.inf

            for ext_fitted in ad_fitted:
                ext_fitted.mask = np.where(
                    (ext_fitted.data.T / row_max).T < threshold,
                    DQ.unilluminated, DQ.good)

            for ext_fitted, indices in zip(ad_fitted, array_info.extensions):
                tiled_arrsec = ext_fitted.array_section()
                for i in indices:
                    ext = ad[i]
                    arrsec = ext.array_section()
                    slice_ = (slice((arrsec.y1 - tiled_arrsec.y1) // ybin, (arrsec.y2 - tiled_arrsec.y1) // ybin),
                              slice((arrsec.x1 - tiled_arrsec.x1) // xbin, (arrsec.x2 - tiled_arrsec.x1) // xbin))
                    ext /= ext_fitted.nddata[slice_]

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)

        return adinputs
