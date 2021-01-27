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
from astropy import table
from astropy import units as u
from scipy.interpolate import UnivariateSpline

from geminidr.core import Spect
from .primitives_gmos import GMOS
from . import parameters_gmos_spect

from geminidr.gemini.lookups import DQ_definitions as DQ
from geminidr.gmos.lookups import geometry_conf as geotable

from gempy.gemini import gemini_tools as gt
from gempy.library import astromodels, astrotools, config, fitting, tracing, transform

from recipe_system.utils.decorators import parameter_override

from ..interactive import server as iserver
from ..interactive.fit import fit1d as ifit1d


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
        "EEV_9273-16-03": [9.883090E-1, -1.390254E-5,  5.282149E-7, -6.847360E-10],
        "EEV_9273-20-03": [9.699E-1, 1.330E-4, -2.082E-7, 1.206E-10],
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
        "EEV_2037-06-03": {"1900-01-01": [2.8197, -8.101e-3, 1.147e-5, -5.270e-9],
                           "2006-08-31": [2.225037, -4.441856E-3, 5.216792E-6, -1.977506E-9]},
        "EEV_8261-07-04": {"1900-01-01": [1.3771, -1.863e-3, 2.559e-6, -1.0289e-9],
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
        a 2D spectral image, based on the wavelength solution of an
        associated processed_arc.

        It is only designed to work on FLATs, and therefore unmosaicked data.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        arc : {None, AstroData, str}
            Arc(s) with distortion map.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        sfx = params["suffix"]
        arc = params["arc"]
        use_iraf = params["use_iraf"]

        # Get a suitable arc frame (with distortion map) for every science AD
        if arc is None:
            self.getProcessedArc(adinputs, refresh=False)
            arc_list = self._get_cal(adinputs, 'processed_arc')
        else:
            arc_list = arc

        for ad, arc in zip(*gt.make_lists(adinputs, arc_list, force_ad=True)):
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by QECorrect".
                            format(ad.filename))
                continue

            if 'e2v' in ad.detector_name(pretty=True):
                log.stdinfo(f"{ad.filename} has the e2v CCDs, so no QE "
                            "correction is necessary")
                continue

            if self.timestamp_keys['mosaicDetectors'] in ad.phu:
                log.warning(f"{ad.filename} has been processed by mosaic"
                            "Detectors so QECorrect cannot be run")
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

            # The plan here is to attach the mosaic gWCS to the science frame,
            # apply an origin shift to put it in the frame of the arc, and
            # then use the arc's WCS to get the wavelength. If there's no arc,
            # we just use the science frame's WCS.
            # Since we're going to change that WCS, store it for restoration.
            original_wcs = [ext.wcs for ext in ad]
            try:
                transform.add_mosaic_wcs(ad, geotable)
            except ValueError:
                log.warning(f"{ad.filename} already has a 'mosaic' coordinate"
                            "frame. This is unexpected but I'll continue.")

            if arc is None:
                if 'sq' in self.mode:
                    raise OSError(f"No processed arc listed for {ad.filename}")
                else:
                    log.warning(f"No arc supplied for {ad.filename}")
            else:
                # OK, we definitely want to try to do this, get a wavelength solution
                if self.timestamp_keys['determineWavelengthSolution'] not in arc.phu:
                    msg = f"Arc {arc.filename} (for {ad.filename} has not been wavelength calibrated."
                    if 'sq' in self.mode:
                        raise IOError(msg)
                    else:
                        log.warning(msg)

                # We'll be modifying this
                arc_wcs = deepcopy(arc[0].wcs)
                if 'distortion_corrected' not in arc_wcs.available_frames:
                    msg = f"Arc {arc.filename} (for {ad.filename}) has no distortion model."
                    if 'sq' in self.mode:
                        raise OSError(msg)
                    else:
                        log.warning(msg)

                # NB. At this point, we could have an arc that has no good
                # wavelength solution nor distortion correction. But we will
                # use its WCS rather than the science frame's because it must
                # have been supplied by the user.

                # This is GMOS so no need to be as generic as distortionCorrect
                ad_detsec = ad.detector_section()
                arc_detsec = arc.detector_section()[0]
                if (ad_detsec[0].x1, ad_detsec[-1].x2) != (arc_detsec.x1, arc_detsec.x2):
                    raise ValueError("I don't know how to process the "
                                     f"offsets between {ad.filename} "
                                     f"and {arc.filename}")

                yoff1 = arc_detsec.y1 - ad_detsec[0].y1
                yoff2 = arc_detsec.y2 - ad_detsec[0].y2
                arc_ext_shapes = [(ext.shape[0] - yoff1 + yoff2,
                                   ext.shape[1]) for ext in ad]
                arc_corners = np.concatenate([transform.get_output_corners(
                    ext.wcs.get_transform(ext.wcs.input_frame, 'mosaic'),
                    input_shape=arc_shape, origin=(yoff1, 0))
                    for ext, arc_shape in zip(ad, arc_ext_shapes)], axis=1)
                arc_origin = tuple(np.ceil(min(corners)) for corners in arc_corners)

                # So this is what was applied to the ARC to get the
                # mosaic frame to its pixel frame, in which the distortion
                # correction model was calculated. Convert coordinates
                # from python order to Model order.
                origin_shift = reduce(Model.__and__, [models.Shift(-origin)
                                                      for origin in arc_origin[::-1]])
                arc_wcs.insert_transform(arc_wcs.input_frame, origin_shift, after=True)

            array_info = gt.array_information(ad)
            if array_info.detector_shape == (1, 3):
                ccd2_indices = array_info.extensions[1]
            else:
                raise ValueError(f"{ad.filename} does not have 3 separate detectors")

            for index, ext in enumerate(ad):
                if index in ccd2_indices:
                    continue

                # Use the WCS in the extension if we don't have an arc,
                # otherwise use the arc's mosaic->world transformation
                if arc is None:
                    trans = ext.wcs.forward_transform
                else:
                    trans = (ext.wcs.get_transform(ext.wcs.input_frame, 'mosaic') |
                             arc_wcs.forward_transform)

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

            for ext, orig_wcs in zip(ad, original_wcs):
                ext.wcs = orig_wcs

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
        use_second_order = w2 > 1000 and abs(dw) < 0.2
        use_second_order = False
        lookup_dir = os.path.dirname(import_module('.__init__', self.inst_lookups).__file__)
        filename = os.path.join(lookup_dir,
                                'CuAr_GMOS{}.dat'.format('_mixord' if use_second_order else ''))
        wavelengths = np.loadtxt(filename, usecols=[0])
        return wavelengths, None

    def traceApertures(self, adinputs=None, **params):
        """
        Traces apertures listed in the `.APERTURE` table along the dispersion
        direction, and estimates the optimal extraction aperture size from the
        spatial profile of each source.

        This primitive is now designed to run on tiled and mosaicked data so
        normal longslit spectra will be in a single extension. We keep the loop
        over extensions to allow the possibility of expanding it to cases where
        we have multiple extensions (e.g. Multi-Object Spectroscopy).

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            Science data as 2D spectral images with a `.APERTURE` table attached
            to one or more of its extensions.
        suffix : str
            Suffix to be added to output files.
        order : int
            Fitting order along spectrum. Default: 2
        step : int
            Step size for sampling along dispersion direction. Default: 10
        nsum : int
            Number of rows/columns to combine at each step. Default: 10
        max_missed : int
            Maximum number of interactions without finding line before line is
            considered lost forever. Default: 5
        max_shift : float
            Maximum perpendicular shift (in pixels) from pixel to pixel.
            Default: 0.05
        debug: bool
            draw aperture traces on image display window?

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Science data as 2D spectral images with the `.APERTURE` the updated
            to contain its upper and lower limits.

        See Also
        --------
        :meth:`~geminidr.core.primitives_spect.Spect.findSourceApertures`

        """
        # Setup log
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        # Parse parameters
        debug = params["debug"]
        is_interactive = params["interactive"]
        max_missed = params["max_missed"]
        max_shift = params["max_shift"]
        nsum = params["nsum"]
        sfx = params["suffix"]
        step = params["step"]

        fit1d_params = fitting.fit_1D.translate_params(
            {**params, "function": "chebyshev"})

        # Main Loop
        for ad in adinputs:
            for ext in ad:

                # Verify inputs
                try:
                    aptable = ext.APERTURE
                    locations = aptable['c0'].data
                except (AttributeError, KeyError):
                    log.warning("Could not find aperture locations in "
                                f"{ad.filename} extension {ext.id} - continuing")
                    continue

                if debug:
                    self.viewer.display_image(ext, wcs=False)
                    self.viewer.width = 2
                    self.viewer.color = "blue"

                dispaxis = 2 - ext.dispersion_axis()  # python sense

                if is_interactive:

                    all_initial_fitting_pars = list()
                    all_domain = list()
                    for ap in ext.APERTURE:
                        all_initial_fitting_pars.append(fit1d_params)
                        all_domain.append([ap['domain_start'], ap['domain_end']])

                    def _get_tracing_knots(conf, extra):
                        """
                        This function is used by the interactive fitter to
                        generate the x,y,weights to use for each fit.

                        Parameters
                        ----------
                        conf : ???
                            ???
                        extra : ???
                            ???

                        Returns
                        -------
                        """
                        all_tracing_knots = list()

                        for _i, _loc in enumerate(locations):
                            _c0 = int(_loc + 0.5)

                            _spectrum = ext.data[_c0] \
                                if dispaxis == 1 else ext.data[:, c0]
                            _start = np.argmax(
                                astrotools.boxcar(_spectrum, size=3))

                            # _, _in_coords = tracing.trace_lines(
                            #     ext, axis=dispaxis, cwidth=5,
                            #     initial=[_loc], initial_tolerance=None,
                            #     max_missed=extra['max_missed'],
                            #     max_shift=extra['max_shift'],
                            #     nsum=extra['nsum'], rwidth=None, start=_start,
                            #     step=extra['step'],
                            #     viewer=self.viewer if debug else None)

                            _, _in_coords = tracing.trace_lines(
                                ext, axis=dispaxis, cwidth=5,
                                initial=[_loc], initial_tolerance=None,
                                max_missed=max_missed,
                                max_shift=max_shift,
                                nsum=nsum, rwidth=None, start=_start,
                                step=step,
                                viewer=self.viewer if debug else None)

                            _in_coords = np.ma.masked_array(_in_coords)

                            # ToDo: This should not be required
                            _in_coords.mask = np.zeros_like(_in_coords)

                            spectral_tracing_knots = _in_coords[1 - dispaxis]
                            spatial_tracing_knots = _in_coords[dispaxis]

                            all_tracing_knots.append([spectral_tracing_knots,
                                                      spatial_tracing_knots])

                        return all_tracing_knots

                    # Pass the primitive configuration to the interactive object.
                    _config = self.params[self.myself()]
                    _config.update(**params)

                    # Create parameters to add to the UI
                    reinit_params = ["max_missed", "max_shift", "nsum", "step"]
                    reinit_extras = {
                        "max_missed": config.RangeField("Max Missed", int, 5, min=0),
                        "max_shift": config.RangeField("Max Shifted", float, 0.05, min=0.001, max=0.1),
                        "nsum": config.RangeField("Number of lines to sum", int, 10, min=1),
                        "step": config.RangeField("Tracing step: ", int, 10, min=1),
                    }

                    # ToDo: Fit1DVisualizer breaks if reinit_extras is None and
                    #  reinit_params is not.
                    visualizer = ifit1d.Fit1DVisualizer(
                        _get_tracing_knots,
                        config=_config,
                        fitting_parameters=all_initial_fitting_pars,
                        tab_name_fmt="Aperture {}",
                        xlabel='x',
                        ylabel='y',
                        reinit_live=True,
                        reinit_params=reinit_params,
                        reinit_extras=reinit_extras,
                        domains=all_domain,
                        title="Trace Apertures")

                    iserver.interactive_fitter(visualizer)

                else:
                    # pop "order" seeing we may need to call fit_1D with a
                    #  different value
                    order = fit1d_params.pop("order")

                    # For efficiency, we would like to trace all sources
                    #  simultaneously (like we do with arc lines), but we need
                    #  to start somewhere the source is bright enough, and there
                    #  may not be a single location where that is true for all
                    #  sources
                    for i, loc in enumerate(locations):
                        c0 = int(loc + 0.5)
                        spectrum = ext.data[c0] if dispaxis == 1 else ext.data[:, c0]
                        start = np.argmax(astrotools.boxcar(spectrum, size=3))

                        # The coordinates are always returned as (x-coords, y-coords)
                        ref_coords, in_coords = tracing.trace_lines(ext, axis=dispaxis,
                                                                    start=start, initial=[loc],
                                                                    rwidth=None, cwidth=5, step=step,
                                                                    nsum=nsum, max_missed=max_missed,
                                                                    initial_tolerance=None,
                                                                    max_shift=max_shift,
                                                                    viewer=self.viewer if debug else None)
                        if i:
                            all_ref_coords = np.concatenate((all_ref_coords, ref_coords), axis=1)
                            all_in_coords = np.concatenate((all_in_coords, in_coords), axis=1)
                        else:
                            all_ref_coords = ref_coords
                            all_in_coords = in_coords

                    spectral_coords = np.arange(0, ext.shape[dispaxis], step)
                    all_tables = []
                    for aperture in aptable:
                        location = aperture['c0']
                        # Funky stuff to extract the traced coords associated with
                        # each aperture (there's just a big list of all the coords
                        # from all the apertures) and sort them by coordinate
                        # along the spectrum
                        coords = np.array([list(c1) + list(c2)
                                           for c1, c2 in zip(all_ref_coords.T, all_in_coords.T)
                                           if c1[dispaxis] == location])
                        values = np.array(sorted(coords, key=lambda c: c[1 - dispaxis])).T
                        ref_coords, in_coords = values[:2], values[2:]

                        # Find model to transform actual (x,y) locations to the
                        # value of the reference pixel along the dispersion axis
                        try:
                            fit1d = fitting.fit_1D(
                                in_coords[dispaxis],
                                domain=[0, ext.shape[dispaxis] - 1],
                                order=order,
                                points=in_coords[1 - dispaxis],
                                **fit1d_params)

                        # This hides a multitude of sins, including no points
                        # returned by the trace, or insufficient points to
                        # constrain fit. We call fit1d with dummy points to
                        # ensure we get the same type of result as if it had
                        # been successful.
                        except (IndexError, np.linalg.linalg.LinAlgError):
                            log.warning(f"Unable to trace aperture {aperture['number']}")
                            fit1d = fitting.fit_1D(
                                np.full_like(spectral_coords, c0),
                                domain=[0, ext.shape[dispaxis] - 1],
                                order=0,
                                points=spectral_coords,
                                **fit1d_params)

                        else:
                            if debug:
                                plot_coords = np.array([spectral_coords, fit1d.evaluate(spectral_coords)]).T
                                self.viewer.polygon(plot_coords, closed=False,
                                                    xfirst=(dispaxis == 1), origin=0)

                        this_aptable = astromodels.model_to_table(fit1d.model)

                        # Recalculate aperture limits after rectification
                        apcoords = fit1d.evaluate(np.arange(ext.shape[dispaxis]))
                        this_aptable["aper_lower"] = aperture["aper_lower"] + (location - apcoords.min())
                        this_aptable["aper_upper"] = aperture["aper_upper"] - (apcoords.max() - location)
                        all_tables.append(this_aptable)

                new_aptable = table.vstack(all_tables, metadata_conflicts="silent")
                ext.APERTURE = new_aptable

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)
        return adinputs
