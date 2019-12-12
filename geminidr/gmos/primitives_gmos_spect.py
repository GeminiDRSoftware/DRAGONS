#
#                                                                  gemini_python
#
#                                                        primtives_gmos_spect.py
# ------------------------------------------------------------------------------
import numpy as np

from importlib import import_module
import os
from datetime import datetime

from geminidr.gemini.lookups import DQ_definitions as DQ
from geminidr.gmos.lookups import geometry_conf as geotable

from gempy.gemini import gemini_tools as gt
from gempy.library import astromodels, transform

from geminidr.core import Spect
from .primitives_gmos import GMOS
from . import parameters_gmos_spect

from astropy.modeling import models
from astropy import units as u

from recipe_system.utils.decorators import parameter_override

# Put this here for now!
def qeModel(ext):
    """

    Parameters
    ----------
    ext: single-slice AstroData object
        the extension to calculate the QE coefficients for

    Returns
    -------
    astropy.modeling.models.Model: the model to convert wavelengths
    """
    qeData = {
        # GMOS-N EEV CCD1 and 3
        "EEV_9273-16-03": [9.883090E-1, -1.390254E-6,  5.282149E-9, -6.847360E-13],
        "EEV_9273-20-03": [9.699E-1, 1.330E-5, -2.082E-9, 1.206E-13],
        # GMOS-N Hamamatsu CCD1 and 3
        "BI13-20-4k-1": [-2.45481760e+03, 3.24130657e+00, -1.87380500e-03,
                         6.23494400e-07, -1.31713482e-10, 1.83308885e-14,
                         -1.68145852e-18, 9.80603592e-23, -3.30016761e-27,
                         4.88466076e-32],
        "BI13-18-4k-2": [3.48333720e+03, -5.27904605e+00, 3.48210500e-03,
                         -1.31286828e-06, 3.12154994e-10, -4.85949692e-14,
                         4.95886638e-18, -3.20198283e-22, 1.18833302e-26,
                         -1.93303639e-31],
        # GMOS-S EEV CCD1 and 3
        "EEV_2037-06-03": {"1900-01-01": [2.8197, -8.101e-4, 1.147e-7, -5.270e-12],
                           "2006-08-31": [2.225037, -4.441856E-4, 5.216792E-8, -1.977506E-12]},
        "EEV_8261-07-04": {"1900-01-01": [1.3771, -1.863e-4, 2.559e-8, -1.0289e-12],
                           "2006-08-31": [8.694583E-1, 1.021462E-4, -2.396927E-8, 1.670948E-12]},
        # GMOS-S Hamamatsu CCD1 and 3
        "BI5-36-4k-2": [-6.00810046e+02,  6.74834788e-01, -3.26251680e-04,
                        8.87677395e-08, -1.48699188e-11, 1.57120033e-15,
                        -1.02326999e-19, 3.75794380e-24, -5.96238257e-29],
        "BI12-34-4k-1": [7.44793105e+02, -1.22941630e+00, 8.83657074e-04,
                         -3.62949805e-07, 9.40246850e-11, -1.59549327e-14,
                         1.77557909e-18, -1.25086490e-22, 5.06582071e-27,
                         -8.99166534e-32]
     }

    array_name = ext.array_name().split(',')[0]
    try:
        coeffs = qeData[array_name]
    except KeyError:
        return None
    if isinstance(coeffs, dict):
        obs_date = ext.ut_date()
        for k in sorted(coeffs):
            if obs_date >= datetime.strptime(k, "%Y-%m-%d"):
                use_coeffs = coeffs[k]
        coeffs = use_coeffs

    model = models.Polynomial1D(degree=len(coeffs)-1)
    for i, c in enumerate(coeffs):
        setattr(model, 'c{}'.format(i), c)
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
            if 'FLAT' not in ad.tags:
                log.warning("{} is not a FLAT so should not be processed by "
                            "applyQECorrection".format(ad.filename))
                continue

            if 'e2v' in ad.detector_name(pretty=True):
                log.warning("{} has the e2v CCDs, so no QE correction "
                            "is necessary".format(ad.filename))
                continue

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
                        qe_correction = qeModel(ext)((waves / u.AA).to(u.dimensionless_unscaled).value)
                    except TypeError:  # qeModel() returns None
                        msg = "No QE correction found for {}:{}".format(ad.filename, ext.hdr['EXTVER'])
                        if 'sq' in self.mode:
                            raise ValueError(msg)
                        else:
                            log.warning(msg)
                    log.fullinfo("Mean relative QE of EXTVER {} is {:.5f}".
                                 format(ext.hdr['EXTVER'], qe_correction.mean()))
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

