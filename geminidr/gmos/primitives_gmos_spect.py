#
#                                                                  gemini_python
#
#                                                        primtives_gmos_spect.py
# ------------------------------------------------------------------------------
import os

import numpy as np
from importlib import import_module
from datetime import datetime
from functools import reduce
from copy import deepcopy

from astropy.modeling import models, Model
from astropy import units as u
from scipy.interpolate import UnivariateSpline

from geminidr.core import Spect
from .primitives_gmos import GMOS
from . import parameters_gmos_spect

from geminidr.gemini.lookups import DQ_definitions as DQ
from geminidr.gmos.lookups import geometry_conf as geotable

from gempy.gemini import gemini_tools as gt
from gempy.library import astromodels as am
from gempy.library import transform, wavecal

from recipe_system.utils.decorators import parameter_override


# Put this here for now!
def qeModel(ext, use_iraf=False):
    """
    This function returns a callable object that returns the QE of a CCD
    (relative to CCD2) as a function of wavelength(s) in nm. The QE data is
    provided as a dict, keyed by the array_name() descriptor of the CCD.
    The value is either a list (interpreted as polynomial coefficients) or a
    dict describing a spline.

    In addition, if the model changes, the value can be a dict keyed by the
    earliest UT date at which each model should be applied.

    Parameters
    ----------
    ext : single-slice AstroData object
        the extension to calculate the QE coefficients for
    use_iraf : bool
        use IRAF fits rather than DRAGONS ones?

    Returns
    -------
    callable: a function to convert wavelengths in nm to relative QE
    """
    # All coefficients are for nm (not AA as in G-IRAF)
    qeData = {
        # GMOS-N EEV CCD1 and 3
        "EEV 9273-16-03": [9.883090E-1, -1.390254E-5,  5.282149E-7, -6.847360E-10],
        "EEV 9273-20-03": [9.699E-1, 1.330E-4, -2.082E-7, 1.206E-10],
        # GMOS-N Hamamatsu CCD1 and 3
        "BI13-20-4k-1": {"order": 3,
                         "knots": [366.5, 413.5, 435.5, 465., 478.5, 507.5, 693., 1062.],
                         "coeffs": [1.20848283, 1.59132929, 1.58317142, 1.25123198, 1.14410563,
                                    0.98095206, 0.83416436, 1.03247587, 1.15355675, 1.10176507]},
        "BI13-18-4k-2": {"order": 3,
                         "knots": [341.75, 389.5, 414., 447.5, 493., 592., 694.5, 1057.],
                         "coeffs": [0.90570141, 0.99834392, 1.6311227 , 1.47271364, 1.13843214,
                                    0.91170917, 0.88454097, 1.06456595, 1.16684561, 1.10476059]},
        # IRAF coefficients
        ("BI13-20-4k-1", "IRAF"): [-2.45481760e+03, 3.24130657e+01, -1.87380500e-01,
                                   6.23494400e-04, -1.31713482e-06, 1.83308885e-09,
                                   -1.68145852e-12, 9.80603592e-16, -3.30016761e-19,
                                   4.88466076e-23],
        ("BI13-18-4k-2", "IRAF"): [3.48333720e+03, -5.27904605e+01, 3.48210500e-01,
                                   -1.31286828e-03, 3.12154994e-06, -4.85949692e-09,
                                   4.95886638e-12, -3.20198283e-15, 1.18833302e-18,
                                   -1.93303639e-22],
        # GMOS-S EEV CCD1 and 3
        "EEV 2037-06-03": {"1900-01-01": [2.8197, -8.101e-3, 1.147e-5, -5.270e-9],
                           "2006-08-31": [2.225037, -4.441856E-3, 5.216792E-6, -1.977506E-9]},
        "EEV 8261-07-04": {"1900-01-01": [1.3771, -1.863e-3, 2.559e-6, -1.0289e-9],
                           "2006-08-31": [8.694583E-1, 1.021462E-3, -2.396927E-6, 1.670948E-9]},
        # GMOS-S Hamamatsu CCD1 and 3
        "BI5-36-4k-2": {"order": 3,
                        "knots": [374., 409., 451., 523.5, 584.5, 733.5, 922., 1070.75],
                        "coeffs": [1.04722893, 0.87968707, 0.70533794, 0.67657144, 0.71217743,
                                   0.82421959, 0.94903734, 1.00847771, 0.98158784, 0.90798127]},
        "BI12-34-4k-1": {"order": 3,
                         "knots": [340.25, 377.5, 406., 439., 511.5, 601., 746., 916.5, 1070.],
                         "coeffs": [0.7433304, 1.07041859, 1.51006315, 1.43997471, 1.03126307,
                                    0.84984109, 0.8944949, 1.02806209, 1.11960524, 1.12224211,
                                    0.95279761]},
        # IRAF coefficients
        ("BI5-36-4k-2", "IRAF"): [-6.00810046e+02,  6.74834788e+00, -3.26251680e-02,
                                  8.87677395e-05, -1.48699188e-07, 1.57120033e-10,
                                  -1.02326999e-13, 3.75794380e-17, -5.96238257e-21],
        ("BI12-34-4k-1", "IRAF"): [7.44793105e+02, -1.22941630e+01, 8.83657074e-02,
                                   -3.62949805e-04, 9.40246850e-07, -1.59549327e-09,
                                   1.77557909e-12, -1.25086490e-15, 5.06582071e-19,
                                   -8.99166534e-23]
    }

    array_name = ext.array_name().split(',')[0]
    key = (array_name, "IRAF") if use_iraf else array_name
    try:
        data = qeData[key]
    except KeyError:
        try:  # fallback for older CCDs where the IRAF solution isn't labelled
            data = qeData[array_name]
        except KeyError:
            return None

    # Deal with date-dependent changes
    if isinstance(data, dict) and 'knots' not in data:
        obs_date = ext.ut_date()
        for k in sorted(data):
            if obs_date >= datetime.strptime(k, "%Y-%m-%d"):
                use_data = data[k]
        data = use_data

    # data is either a dict defining a spline that defines QE
    # or a list of polynomial coefficients that define QE
    if 'knots' in data:
        # Duplicate the knots at either end for the correct format
        order = data["order"]
        knots = data["knots"]
        knots[0:0] = [knots[0]] * order
        knots.extend(knots[-1:] * order)
        coeffs = data["coeffs"] + [0] * (order+1)
        spline = UnivariateSpline._from_tck((knots, coeffs, order))
        return spline
    else:
        model_params = {'c{}'.format(i): c for i, c in enumerate(data)}
        model = models.Polynomial1D(degree=len(data)-1, **model_params)
        return model

# ------------------------------------------------------------------------------
@parameter_override
class GMOSSpect(Spect, GMOS):
    """
    This is the class containing all of the preprocessing primitives
    for the GMOSSpect level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = {"GEMINI", "GMOS", "SPECT"}

    def __init__(self, adinputs, **kwargs):
        super().__init__(adinputs, **kwargs)
        self._param_update(parameters_gmos_spect)

    def QECorrect(self, adinputs=None, **params):
        """
        This primitive applies a wavelength-dependent QE correction to
        a 2D spectral image, based on the wavelength solution in the WCS
        (from `attachWavelengthSolution` or, in non-SQ-modes, the initial
        linear approximation).

        It is only designed to work on FLATs, and therefore unmosaicked data.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        sfx = params["suffix"]
        use_iraf = params["use_iraf"]
        do_cal = params["do_cal"]

        if do_cal == 'skip':
            log.warning("QE correction has been turned off.")
            return adinputs

        for ad in adinputs:

            if ad.phu.get(timestamp_key):
                log.warning(f"{ad.filename}: already processed by QECorrect. "
                            "Continuing.")
                continue

            if 'e2v' in ad.detector_name(pretty=True):
                log.stdinfo(f"{ad.filename} has the e2v CCDs, so no QE "
                            "correction is necessary")
                continue

            if self.timestamp_keys['mosaicDetectors'] in ad.phu:
                msg = (f"{ad.filename} has been processed by mosaicDetectors "
                        "so cannot correct QE for each CCD")
                if 'sq' in self.mode or do_cal == 'force':
                    raise ValueError(msg)
                log.warning(msg)
                continue

            # Determines whether to multiply or divide by QE correction
            is_flat = 'FLAT' in ad.tags

            array_info = gt.array_information(ad)
            if array_info.detector_shape == (1, 3):
                ccd2_indices = array_info.extensions[1]
            else:
                raise ValueError(f"{ad.filename} does not have 3 separate detectors")

            for index, ext in enumerate(ad):
                if index in ccd2_indices:
                    continue

                trans = ext.wcs.forward_transform

                # There should always be a wavelength model (even if it's an
                # approximation) as long as the data have been prepared, but
                # check and produce a clear error if not:
                try:
                    am.get_named_submodel(trans, 'WAVE')
                except (AttributeError, IndexError):
                    raise ValueError('No wavelength solution for '
                                     f'{ad.filename}, extension {ext.id}')

                # For SQ, require that the distortion correction is included,
                # either in the WCS or possibly by prior rectification (though
                # this is a corner case since mosaicking is disallowed). This
                # check might need revisiting if distortion correction gets
                # included in any other resampling steps in future, but by that
                # point we may be propagating an "already applied" WCS (from
                # resampled to raw co-ordinates) that would make it easier.
                if ('distortion_corrected' not in ext.wcs.available_frames and
                      self.timestamp_keys['distortionCorrect'] not in ad.phu):
                    msg = ('No distortion correction in WCS for '
                           f'{ad.filename}, extension {ext.id}')
                    if 'sq' in self.mode:
                        raise ValueError(msg)
                    log.warning(msg)

                ygrid, xgrid = np.indices(ext.shape)
                # TODO: want with_units
                waves = trans(xgrid, ygrid)[0] * u.nm  # Wavelength always axis 0

                # Tapering required to prevent QE correction from blowing up
                # at the extremes (remember, this is a ratio, not the actual QE)
                # We use half-Gaussians to taper
                taper = np.ones_like(ext.data)
                taper_locut, taper_losig = 350 * u.nm, 25 * u.nm
                taper_hicut, taper_hisig = 1200 * u.nm, 200 * u.nm
                taper[waves < taper_locut] = np.exp(-((waves[waves < taper_locut]
                                                       - taper_locut) / taper_losig) ** 2)
                taper[waves > taper_hicut] = np.exp(-((waves[waves > taper_hicut]
                                                       - taper_hicut) / taper_hisig) ** 2)
                try:
                    qe_correction = (qeModel(ext, use_iraf=use_iraf)(
                        (waves / u.nm).to(u.dimensionless_unscaled).value).astype(
                        np.float32) - 1) * taper + 1
                except TypeError:  # qeModel() returns None
                    msg = f"No QE correction found for {ad.filename} extension {ext.id}"
                    if 'sq' in self.mode:
                        raise ValueError(msg)
                    else:
                        log.warning(msg)
                        continue
                log.stdinfo(f"Mean relative QE of extension {ext.id} is "
                            f"{qe_correction.mean():.5f}")
                if not is_flat:
                    qe_correction = 1. / qe_correction
                ext.multiply(qe_correction)

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

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

            shuffle = ad.shuffle_pixels() // ad.detector_y_bin()
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

    def standardizeWCS(self, adinputs=None, suffix=None):
        """
        This primitive updates the WCS attribute of each NDAstroData extension
        in the input AstroData objects. For spectroscopic data, it means
        replacing an imaging WCS with an approximate spectroscopic WCS.

        This is a GMOS-specific primitive due to the systematic offsets for
        GMOS-S at central wavelengths > 950nm.

        Parameters
        ----------
        suffix: str/None
            suffix to be added to output files

        """
        log = self.log
        timestamp_key = self.timestamp_keys[self.myself()]
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        for ad in adinputs:
            log.stdinfo(f"Adding spectroscopic WCS to {ad.filename}")
            cenwave = ad.central_wavelength(asNanometers=True)
            if ad.instrument() == "GMOS-S" and cenwave > 950:
                cenwave += (6.89483617 - 0.00332086 * cenwave) * cenwave - 3555.048
            else:
                cenwave = None
            transform.add_longslit_wcs(ad, central_wavelength=cenwave)

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

    def _get_arc_linelist(self, waves=None):
        use_second_order = waves.max() > 1000 and abs(np.diff(waves).mean()) < 0.2

        use_second_order = False
        lookup_dir = os.path.dirname(import_module('.__init__',
                                                   self.inst_lookups).__file__)
        filename = os.path.join(lookup_dir,
                                'CuAr_GMOS{}.dat'.format('_mixord' if use_second_order else ''))
        return wavecal.LineList(filename)
