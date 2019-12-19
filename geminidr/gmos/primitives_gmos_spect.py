#
#                                                                  gemini_python
#
#                                                        primtives_gmos_spect.py
# ------------------------------------------------------------------------------
import os

import numpy as np
from importlib import import_module
from datetime import datetime

from astropy.modeling import models
from astropy import units as u
from scipy.interpolate import UnivariateSpline

from geminidr.core import Spect
from .primitives_gmos import GMOS
from . import parameters_gmos_spect

from geminidr.gemini.lookups import DQ_definitions as DQ
from geminidr.gmos.lookups import geometry_conf as geotable

from gempy.gemini import gemini_tools as gt
from gempy.library import astromodels, transform

from recipe_system.utils.decorators import parameter_override

# Put this here for now!
def qeModel(ext):
    """
    This function returns a callable object that returns the QE of a CCD
    (relative to CCD2) as a function of wavelength(s) in nm. The QE data is
    provided as a dict, keyed by the array_name() descriptor of the CCD.
    The value is either a list (interpreted as polynomial coefficients) or a
    dict describing a spline.

    NB. Spline objects defined in the dict return the decimal logarithm of
    the relative QE, while polynomials simply return the relative QE.

    In addition, if the model changes, the value can be a dict keyed by the
    earliest UT date at which each model should be applied.

    Parameters
    ----------
    ext: single-slice AstroData object
        the extension to calculate the QE coefficients for

    Returns
    -------
    callable: a function to convert wavelengths in nm to relative QE
    """
    # All coefficients are for nm (not AA as in G-IRAF)
    # Polynomials are simply a list, splines are a dict
    # In addition, an entry can be a dict with keys corresponding to the
    # earliest da
    qeData = {
        # GMOS-N EEV CCD1 and 3
        "EEV_9273-16-03": [9.883090E-1, -1.390254E-5,  5.282149E-7, -6.847360E-10],
        "EEV_9273-20-03": [9.699E-1, 1.330E-4, -2.082E-7, 1.206E-10],
        # GMOS-N Hamamatsu CCD1 and 3
        "BI13-20-4k-1": {"order": 3,
                         "knots": [358.0, 432.0, 476.0, 520.0, 602.0, 640.0, 678.0, 752.0, 914.0, 1002.0, 1040.0, 1076.0],
                         "coeffs": [0.02206208508099143, 0.2834094598715138, 0.07132646057310524, -0.01980030661665999,
                                    -0.05201598712929662, -0.06777120754328926, -0.012413172958416104, -0.015591358664326838,
                                    0.03433933272643748, 0.04127142803163095, 0.06235368833554948, -0.008691968589858072,
                                    0.06049075935311075, 0.0484080146014316]},
        "BI13-18-4k-2": {"order": 3,
                         "knots": [358.0, 380.0, 428.0, 468.0, 576.0, 648.0, 798.0, 932.0, 966.0, 994.0, 1010.0, 1028.0, 1044.0, 1076.0],
                         "coeffs": [0.014629039930259876, 0.10133317072093591, 0.2384728514854233, 0.0778544705196136,
                                    -0.06232251896695778, -0.0671226704286497, 0.0017962888751030116, 0.02399802448657926,
                                    0.06062000896013293, 0.04661836594286457, 0.05694058456700794, 0.020108979328507717,
                                    0.00719658389760285, 0.029938578274652766, 0.05265151968369216, 0.04654560999567498]},
        # GMOS-S EEV CCD1 and 3
        "EEV_2037-06-03": {"1900-01-01": [2.8197, -8.101e-3, 1.147e-5, -5.270e-9],
                           "2006-08-31": [2.225037, -4.441856E-3, 5.216792E-6, -1.977506E-9]},
        "EEV_8261-07-04": {"1900-01-01": [1.3771, -1.863e-3, 2.559e-6, -1.0289e-9],
                           "2006-08-31": [8.694583E-1, 1.021462E-3, -2.396927E-6, 1.670948E-9]},
        # GMOS-S Hamamatsu CCD1 and 3
        "BI5-36-4k-2": [-6.00810046e+02,  6.74834788e+00, -3.26251680e-02,
                        8.87677395e-05, -1.48699188e-07, 1.57120033e-10,
                        -1.02326999e-13, 3.75794380e-17, -5.96238257e-21],
        "BI12-34-4k-1": [7.44793105e+02, -1.22941630e+01, 8.83657074e-02,
                         -3.62949805e-04, 9.40246850e-07, -1.59549327e-09,
                         1.77557909e-12, -1.25086490e-15, 5.06582071e-19,
                         -8.99166534e-23]
     }

    array_name = ext.array_name().split(',')[0]
    try:
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

    # data is either a dict defining a spline that defines log10(QE)
    # or a list of polynomial coefficients that define QE
    if 'knots' in data:
        # Duplicate the knots at either end for the correct format
        order = data["order"]
        knots = data["knots"]
        knots[0:0] = [knots[0]] * order
        knots.extend(knots[-1:] * order)
        coeffs = data["coeffs"] + [0] * (order+1)
        spline = UnivariateSpline._from_tck((knots, coeffs, order))
        return lambda x: 10 ** spline(x)
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
    tagset = set(["GEMINI", "GMOS", "SPECT"])

    def __init__(self, adinputs, **kwargs):
        super(GMOSSpect, self).__init__(adinputs, **kwargs)
        self._param_update(parameters_gmos_spect)

    def applyQECorrection(self, adinputs=None, **params):
        """
        This primitive applies a wavelength-dependent QE correction to
        a 2D spectral image, based on the wavelength solution of an
        associated processed_arc.

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
        arc = params["arc"]

        # Get a suitable arc frame (with distortion map) for every science AD
        if arc is None:
            self.getProcessedArc(adinputs, refresh=False)
            arc_list = self._get_cal(adinputs, 'processed_arc')
        else:
            arc_list = arc

        distort_model = models.Identity(2)

        for ad, arc in zip(*gt.make_lists(adinputs, arc_list, force_ad=True)):
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by applyQECorrection".
                            format(ad.filename))
                continue

            if 'e2v' in ad.detector_name(pretty=True):
                log.warning("{} has the e2v CCDs, so no QE correction "
                            "is necessary".format(ad.filename))
                continue

            # Determines whether to multiply or divide by QE correction
            is_flat = 'FLAT' in ad.tags

            # If the arc's binning doesn't match, we may still be able to
            # fall back to the approximate solution
            xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()
            if arc is not None and (arc.detector_x_bin() != xbin or
                                    arc.detector_y_bin() != ybin):
                log.warning("Science frame {} and arc {} have different binnings,"
                            "so cannot use arc".format(ad.filename, arc.filename))
                arc = None

            # OK, we definitely want to try to do this, get a wavelength solution
            try:
                wavecal = arc[0].WAVECAL
            except (TypeError, AttributeError):
                wave_model = None
            else:
                model_dict = dict(zip(wavecal['name'], wavecal['coefficients']))
                wave_model = astromodels.dict_to_chebyshev(model_dict)
                if not isinstance(wave_model, models.Chebyshev1D):
                    log.warning("Problem reading wavelength solution from arc "
                                "{}".format(arc.filename))

            if wave_model is None:
                if 'sq' in self.mode:
                    raise IOError("No wavelength solution for {}".format(ad.filename))
                else:
                    log.warning("Using approximate wavelength solution for "
                                "{}".format(ad.filename))

            try:
                fitcoord = arc[0].FITCOORD
            except (TypeError, AttributeError):
                # distort_model already has Identity inverse so nothing required
                pass
            else:
                # TODO: This is copied from determineDistortion() and will need
                # to be refactored out. Or we might be able to simply replace it
                # with a gWCS.pixel_to_world() call
                model_dict = dict(zip(fitcoord['inv_name'],
                                      fitcoord['inv_coefficients']))
                m_inverse = astromodels.dict_to_chebyshev(model_dict)
                if not isinstance(m_inverse, models.Chebyshev2D):
                    log.warning("Problem reading distortion model from arc "
                                "{}".format(arc.filename))
                else:
                    distort_model.inverse = models.Mapping((0, 1, 1)) | (m_inverse & models.Identity(1))

            if distort_model.inverse == distort_model:  # Identity(2)
                if 'sq' in self.mode:
                    raise IOError("No distortion model for {}".format(ad.filename))
                else:
                    log.warning("Proceeding without a disortion correction for "
                                "{}".format(ad.filename))

            ad_detsec = ad.detector_section()
            adg = transform.create_mosaic_transform(ad, geotable)
            if arc is not None:
                arc_detsec = arc.detector_section()[0]
                shifts = [c1 - c2 for c1, c2 in zip(np.array(ad_detsec).min(axis=0),
                                                    arc_detsec)]
                xshift, yshift = shifts[0] / xbin, shifts[2] / ybin  # x1, y1
                if xshift or yshift:
                    log.stdinfo("Found a shift of ({},{}) pixels between "
                                "{} and the calibration.".
                                format(xshift, yshift, ad.filename))
                add_shapes, add_transforms = [], []
                for (arr, trans) in adg:
                    # Try to work out shape of this Block in the unmosaicked
                    # arc, and then apply a shift to align it with the
                    # science Block before applying the same transform.
                    if xshift == 0:
                        add_shapes.append(((arc_detsec.y2 - arc_detsec.y1) // ybin, arr.shape[1]))
                    else:
                        add_shapes.append((arr.shape[0], (arc_detsec.x2 - arc_detsec.x1) // xbin))
                    t = transform.Transform(models.Shift(-xshift) & models.Shift(-yshift))
                    t.append(trans)
                    add_transforms.append(t)
                adg.calculate_output_shape(additional_array_shapes=add_shapes,
                                           additional_transforms=add_transforms)
                origin_shift = models.Shift(-adg.origin[1]) & models.Shift(-adg.origin[0])
                for t in adg.transforms:
                    t.append(origin_shift)

            # Irrespective of arc or not, apply the distortion model (it may
            # be Identity), recalculate output_shape and reset the origin
            for t in adg.transforms:
                t.append(distort_model.copy())
            adg.calculate_output_shape()
            adg.reset_origin()

            # Now we know the shape of the output, we can construct the
            # approximate wavelength solution; ad.dispersion() returns a list!
            if wave_model is None:
                wave_model = (models.Shift(-0.5 * adg.output_shape[1]) |
                              models.Scale(ad.dispersion(asNanometers=True)[0]) |
                              models.Shift(ad.central_wavelength(asNanometers=True)))

            for ccd, (block, trans) in enumerate(adg, start=1):
                if ccd == 2:
                    continue
                for ext, corner in zip(block, block.corners):
                    ygrid, xgrid = np.indices(ext.shape)
                    xgrid += corner[1]  # No need for ygrid
                    xnew = trans(xgrid, ygrid)[0]
                    # Some unit-based stuff here to prepare for gWCS
                    waves = wave_model(xnew) * u.nm
                    try:
                        qe_correction = qeModel(ext)((waves / u.nm).to(u.dimensionless_unscaled).value)
                    except TypeError:  # qeModel() returns None
                        msg = "No QE correction found for {}:{}".format(ad.filename, ext.hdr['EXTVER'])
                        if 'sq' in self.mode:
                            raise ValueError(msg)
                        else:
                            log.warning(msg)
                    #qe_correction[qe_correction < 0] = 0
                    log.fullinfo("Mean relative QE of EXTVER {} is {:.5f}".
                                 format(ext.hdr['EXTVER'], qe_correction.mean()))
                    if not is_flat:
                        qe_correction = 1. / qe_correction
                    qe_correction[qe_correction < 0] = 0
                    qe_correction[qe_correction > 10] = 0
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

    def _get_arc_linelist(self, ext, w1=None, w2=None, dw=None):
        use_second_order = w2 > 820 and abs(dw) < 0.2
        use_second_order = False
        lookup_dir = os.path.dirname(import_module('.__init__', self.inst_lookups).__file__)
        filename = os.path.join(lookup_dir,
                                'CuAr_GMOS{}.dat'.format('_mixord' if use_second_order else ''))

        return np.loadtxt(filename, usecols=[0]), None

