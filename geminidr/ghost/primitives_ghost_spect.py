#
#                                                                  gemini_python
#
#                                                      primitives_ghost_spect.py
# ------------------------------------------------------------------------------
import os
import numpy as np
import math
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
import astropy.coordinates as astrocoord
from astropy.time import Time
from astropy.io.ascii.core import InconsistentTableError
from astropy import units as u
from astropy import constants as const
from astropy.stats import sigma_clip
from astropy.modeling import fitting, models
from astropy.table import Table
from scipy import interpolate
from scipy.ndimage import measurements
from gwcs import coordinate_frames as cf
from gwcs.wcs import WCS as gWCS
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from importlib import import_module

import astrodata
from astrodata import wcs as adwcs
from astrodata.provenance import add_provenance
from gemini_instruments.gemini import get_specphot_name

from geminidr.core.primitives_spect import Spect
from geminidr.gemini.lookups import DQ_definitions as DQ
from geminidr.gemini.lookups import extinction_data as extinct
from geminidr import CalibrationNotFoundError
from gempy.library.nddops import NDStacker
from gempy.library import peak_finding, transform, astrotools as at
from gempy.adlibrary.manipulate_ad import rebin_data
from gempy.gemini import gemini_tools as gt

from .polyfit import GhostArm, Extractor, SlitView
from .polyfit.polyspect import WaveModel

from .primitives_ghost import GHOST, filename_updater

from . import parameters_ghost_spect

from .lookups import polyfit_lookup, line_list

from recipe_system.utils.decorators import parameter_override, capture_provenance
from recipe_system.utils.md5 import md5sum
# ------------------------------------------------------------------------------

GEMINI_SOUTH_LOC = astrocoord.EarthLocation.from_geodetic(
    '-70:44:12.096 degrees', '-30:14:26.700 degrees',
    height=2722., ellipsoid='WGS84'
)

BAD_FLAT_FLAG = 16

# FIXME: This should go somewhere else, but where?
from scipy.ndimage import median_filter
def convolve_with_mask(data, mask, rectangle_width = (100,20)):
    """Helper function to convolve a masked array with a uniform rectangle after median
    filtering to remove cosmic rays.
    """
    #Create our rectangular function
    rectangle_function = np.zeros_like(data)
    rectangle_function[:rectangle_width[0], :rectangle_width[1]] = 1.0
    rectangle_function = np.roll(rectangle_function, int(-rectangle_width[
        0] / 2), axis=0)
    rectangle_function = np.roll(rectangle_function, int(-rectangle_width[1]/2),
                                 axis=1)
    rectangle_fft = np.fft.rfft2(rectangle_function)

    #Median filter in case of cosmic rays
    filt_data = median_filter(data,3)

    #Now convolve. The mask is never set to exactly zero in order to avoid divide
    #by zero errors outside the mask.
    convolved_data = np.fft.irfft2(np.fft.rfft2(filt_data * (mask + 1e-4))*rectangle_fft)
    convolved_data /= np.fft.irfft2(np.fft.rfft2(mask + 1e-4)*rectangle_fft)
    return convolved_data


@parameter_override
@capture_provenance
class GHOSTSpect(GHOST):
    """
    Primitive class for processing GHOST science data.

    This class contains the primitives necessary for processing GHOST science
    data, as well as all related calibration files from the main spectrograph
    cameras. Slit viewer images are processed with another primitive class
    (:class:`ghostdr.ghost.primitives_ghost_slit.GHOSTSlit`).
    """

    """Applicable tagset"""
    tagset = set(["GEMINI", "GHOST"])  # NOT SPECT because of bias/dark

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_ghost_spect)

    def attachWavelengthSolution(self, adinputs=None, **params):
        """
        Compute and append a wavelength solution for the data.

        The GHOST instrument is designed to be very stable over a long period
        of time, so it is not strictly necessary to take arcs for every
        observation. The alternative is use the arcs taken most recently
        before and after the observation of interest, and compute an
        average of their wavelength solutions.

        The average is weighted by
        the inverse of the time between each arc observation and science
        observation. E.g., if the 'before' arc is taken 12 days before the
        science observation, and the 'after' arc is taken 3 days after the
        science observation, then the 'after' arc will have a weight of 80%
        in the final wavelength solution (12/15), and the 'before' arc 20%
        (3/15).

        In the event that either a 'before' arc can't be found but an 'after'
        arc can, or vice versa, the wavelength solution from the arc that was
        found will be applied as-is. If neither a 'before' nor 'after' arc can
        be found, an IOError will be raised.

        It is possible to explicitly pass which arc files to use as
        the ``arc`` parameter. This should be a list of two-tuples, with each
        tuple being of the form
        ``('before_arc_filepath', 'after_arc_filepath')``. This list must be
        the same length as the list of ``adinputs``, with a one-to-one
        correspondence between the two lists.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        arc: list of two-tuples
            A list of two-tuples, with each tuple corresponding to an element of
            the ``adinputs`` list. Within each tuple, the two elements are the
            designated 'before' and 'after' arc for that observation.
        """

        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        # No attempt to check if this primitive has already been run

        # CJS: Heavily edited because of the new AD way
        # Get processed slits, slitFlats, and flats (for xmod)
        # slits and slitFlats may be provided as parameters
        # arc_list = params["arcs"]
        arc_before_file = params["arc_before"]
        arc_after_file = params["arc_after"]
        if arc_before_file:
            arc_before = astrodata.open(arc_before_file)
        if arc_after_file:
            arc_after = astrodata.open(arc_after_file)

        input_frame = adwcs.pixel_frame(2)
        output_frame = cf.SpectralFrame(axes_order=(0,), unit=u.nm,
                                        axes_names=("AWAV",),
                                        name="Wavelength in air")
        # for ad, arcs in zip(
        #         *gt.make_lists(adinputs, arc_list, force_ad=True)):
        for i, ad in enumerate(adinputs):

            if arc_before_file is None and arc_after_file is None:
                # Fetch the arc_before and arc_after in sequence
                arc_before = self._request_bracket_arc(ad, before=True)
                arc_after = self._request_bracket_arc(ad, before=False)

            if arc_before is None and arc_after is None:
                raise CalibrationNotFoundError('No valid arcs found for '
                                               f'{ad.filename}')

            log.stdinfo(f'Arcs for {ad.filename}:')
            if arc_before:
                log.stdinfo(f'   before: {arc_before.filename}')
            if arc_after:
                log.stdinfo(f'    after: {arc_after.filename}')

            # Stand up a GhostArm instance for this ad
            gs = GhostArm(arm=ad.arm(), mode=ad.res_mode(),
                          detector_x_bin=ad.detector_x_bin(),
                          detector_y_bin=ad.detector_y_bin())

            if arc_before is None:
                wfit = gs.evaluate_poly(arc_after[0].WFIT)
                ad.phu.set('ARCIM_A', os.path.abspath(arc_after.filename),
                           "'After' arc image")
            elif arc_after is None:
                wfit = gs.evaluate_poly(arc_before[0].WFIT)
                ad.phu.set('ARCIM_B', os.path.abspath(arc_before.filename),
                           "'Before' arc image")
            else:
                # Need to weighted-average the wavelength fits from the arcs
                # Determine the weights (basically, the inverse time between
                # the observation and the arc)
                wfit_b = gs.evaluate_poly(arc_before[0].WFIT)
                wfit_a = gs.evaluate_poly(arc_after[0].WFIT)
                weight_b = np.abs((arc_before.ut_datetime() -
                                   ad.ut_datetime()).total_seconds())
                weight_a = np.abs((arc_after.ut_datetime() -
                                   ad.ut_datetime()).total_seconds())
                weight_a, weight_b = 1. / weight_a, 1 / weight_b
                log.stdinfo('Cominbing wavelength solutions with weights '
                            '%.3f, %.3f' %
                            (weight_a / (weight_a + weight_b),
                             weight_b / (weight_a + weight_b),
                             ))
                # Compute weighted mean fit
                wfit = wfit_a * weight_a + wfit_b * weight_b
                wfit /= (weight_a + weight_b)
                ad.phu.set('ARCIM_A', os.path.abspath(arc_after.filename),
                           self.keyword_comments['ARCIM_A'])
                ad.phu.set('ARCIM_B', os.path.abspath(arc_before.filename),
                           self.keyword_comments['ARCIM_B'])
                ad.phu.set('ARCWT_A', weight_a,
                           self.keyword_comments['ARCWT_A'])
                ad.phu.set('ARCWT_B', weight_b,
                           self.keyword_comments['ARCWT_B'])

            # rebin the wavelength fit to match the rest of the extensions
            for _ in range(int(math.log(ad.detector_x_bin(), 2))):
                wfit = wfit[:, ::2] + wfit[:, 1::2]
                wfit /= 2.0

            for ext in ad:
                # Needs to be transposed because of astropy x-first
                # set bounding_box=None to avoid GwcsBoundingBoxWarning later
                ext.wcs = gWCS([(input_frame, models.Tabular2D(
                    lookup_table=0.1 * wfit.T, name="WAVE", bounding_box=None)),
                                (output_frame, None)])

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)
            if arc_before:
                add_provenance(ad, arc_before.filename, md5sum(arc_before.path) or "", self.myself())
            if arc_after:
                add_provenance(ad, arc_after.filename, md5sum(arc_after.path) or "", self.myself())

        return adinputs

    def barycentricCorrect(self, adinputs=None, **params):
        """
        Perform barycentric correction of the wavelength extension in the input
        files.

        Barycentric correction is performed by multiplying the wavelength
        scale by a correction factor based on the radial velocity. The velocity
        scan be upplied manually, or can be left to be calculated based on the
        headers in the AstroData input.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        correction_factor: float
            Barycentric correction factor to be applied. Defaults to None, at
            which point a computed value will be applied. The computed value
            is based on the recorded position of the Gemini South observatory.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        velocity = params["velocity"]

        if velocity == 0.0:
            log.stdinfo("A radial velocity of 0.0 has been provided - no "
                        "barycentric correction will be applied")
            return adinputs

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by barycentricCorrect".
                            format(ad.filename))
                continue

            # Get or compute the correction factor
            if velocity is None:
                ra, dec = ad.ra(), ad.dec()
                if ra is None or dec is None:
                    log.warnings("Unable to compute barycentric correction for"
                                 f" {ad.filename} (no sky pos data) - skipping")
                    rv = None
                else:
                    self.log.stdinfo(f"Computing SkyCoord for {ra}, {dec}")
                    sc = astrocoord.SkyCoord(ra, dec, unit=(u.deg, u.deg,))

                    # Compute central time of observation
                    dt_midp = ad.ut_datetime() + timedelta(
                        seconds=ad.exposure_time() / 2.0)
                    dt_midp = Time(dt_midp)
                    bjd = dt_midp.jd1 + dt_midp.jd2

                    # Vanilla AstroPy Implementation
                    rv = sc.radial_velocity_correction(
                        'barycentric', obstime=dt_midp,
                        location=GEMINI_SOUTH_LOC).to(u.km / u.s)
            else:
                rv = velocity * u.km / u.s
                bjd = None

            if rv is not None:
                log.stdinfo("Applying radial velocity correction of "
                            f"{rv.value} km/s to {ad.filename}")
                cf = float(1 + rv / const.c)  # remove u.dimensionless_unscaled
                # We'd really like to just append a Scale(cf) to the WAVE model
                # but our gwcs_to_fits() can't handle that.
                # Even though the wcs is only a single wavelength model, write
                # fairly generic code here in case it becomes reusable
                for ext in ad:
                    wcs_transform = ext.wcs.forward_transform
                    try:
                        wave_model = ext.wcs.forward_transform.get_named_submodel("WAVE")
                    except AttributeError:
                        wave_model = wcs_transform
                    if wave_model.n_submodels == 1:
                        wave_model = wave_model | models.Identity(1)
                    for m in wave_model:
                        if isinstance(m, (models.Tabular1D, models.Tabular2D)):
                            m.lookup_table *= cf
                            break
                        elif isinstance(m, models.Exponential1D):
                            m.amplitude *= cf
                            break
                    else:
                        ndim = ext.wcs.world_n_dim
                        if ndim == 1:
                            ext.wcs.insert_transform(ext.wcs.output_frame, models.Scale(cf))
                        else:
                            ext.wcs.insert_transform(ext.wcs.output_frame,
                                                     models.Scale(cf) | models.Identity(ndim-1))

                # Only one correction per AD right now
                ad.hdr['BERV'] = (rv.value, "Barycentric correction applied (km s-1)")
                if bjd is not None:
                    ad.hdr['BJD'] = (bjd, "Barycentric Julian date")

                # Timestamp and update filename
                gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
                ad.update_filename(suffix=params["suffix"], strip=True)

        return adinputs

    def calculateSensitivity(self, adinputs=None, **params):
        """
        Calculates the overall sensitivity of the observation system
        (instrument, telescope, detector, etc) for each wavelength using
        spectrophotometric data. It is obtained using the ratio
        between the observed data and the reference look-up data.

        Parameters
        ----------
        suffix :  str, optional
            Suffix to be added to output files
        filename: str or None, optional
            Location of spectrophotometric data file. If it is None, uses
            look up data based on the object name stored in OBJECT header key
            (default).
        order : int
            Order of the spline fit to be performed
        in_vacuo: bool
            Are the wavelengths in the spectrophotometric datafile in vacuo?
        debug_plots: bool
            Output some helpful(?) plots
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        datafile = params["filename"]
        poly_degree = params["order"]
        in_vacuo = params["in_vacuo"]
        debug_plots = params["debug_plots"]

        # TODO: consider how to get the same units as core method?
        sensfunc_units = "W m-2 nm-1"

        # We're going to look in the generic (gemini) module as well as the
        # instrument module, so define that
        module = self.inst_lookups.split('.')
        module[-2] = 'gemini'
        gemini_lookups = '.'.join(module)

        for ad in adinputs:
            if datafile is None:
                specphot_name = get_specphot_name(ad)
                if specphot_name is None:
                    specphot_name = ad.object().lower().replace(' ', '')
                filename = f'{specphot_name}.dat'
                for module in (self.inst_lookups, gemini_lookups, 'geminidr.core.lookups'):
                    try:
                        path = import_module('.', module).__path__[0]
                    except (ImportError, ModuleNotFoundError):
                        continue
                    full_path = os.path.join(path, 'spectrophotometric_standards', filename)
                    try:
                        spec_table = Spect._get_spectrophotometry(
                            self, full_path, in_vacuo=in_vacuo)
                    except (FileNotFoundError, InconsistentTableError):
                        pass
                    else:
                        log.stdinfo(f"{ad.filename}: Using spectrophotometric "
                                    f"data file {full_path}")
                        break
                else:
                    log.warning("Cannot read spectrophotometric data table. "
                                "Unable to determine sensitivity for {}".
                                format(ad.filename))
                    continue
            else:
                try:
                    spec_table = Spect._get_spectrophotometry(
                        self, datafile, in_vacuo=in_vacuo)
                except FileNotFoundError:
                    log.warning(f"Cannot find spectrophotometric data table {datafile}."
                                f" Unable to determine sensitivity for {ad.filename}")
                    continue
                except InconsistentTableError:
                    log.warning(f"Cannot read spectrophotometric data table {datafile}."
                                f" Unable to determine sensitivity for {ad.filename}")
                    continue
                else:
                    log.stdinfo(f"{ad.filename}: Using spectrophotometric "
                                f"data file {datafile} as supplied by user")

            spectral_bin = ad.detector_x_bin()
            target = 0  # according to new extractSpectra() behaviour
            ext = ad[target]
            if ext.hdr.get('BUNIT') != "electron":
                raise ValueError(f"{ad.filename} is not in units of electron")

            if "AWAV" in ext.wcs.output_frame.axes_names:
                wavecol_name = "WAVELENGTH_AIR"
                log.stdinfo(f"{ad.filename} is calibrated to air wavelengths")
            elif "WAVE" in ext.wcs.output_frame.axes_names:
                wavecol_name = "WAVELENGTH_VACUUM"
                log.stdinfo(f"{ad.filename} is calibrated to vacuum wavelengths")
            else:
                raise ValueError("Cannot interpret wavelength scale "
                                 f"for {ad.filename}")

            # Re-grid the standard reference spectrum onto the wavelength grid of
            # the observed standard
            regrid_std_ref = np.zeros_like(ext.data)
            waves = make_wavelength_table(ext)
            for od in range(ext.shape[0]):
                #regrid_std_ref[od] = self._regrid_spect(
                #    spec_table['FLUX'].value,
                #    spec_table[wavecol_name].to(u.nm).value,
                #    waves[od],
                #    waveunits='nm'
                #)
                regrid_std_ref[od] = np.interp(
                    waves[od], spec_table[wavecol_name].to(u.nm).value,
                    spec_table['FLUX'].value, left=0, right=0)

            regrid_std_ref = (regrid_std_ref * spec_table['FLUX'].unit).to(
                sensfunc_units, equivalencies=u.spectral_density(waves * u.nm)
            ).value

            # Compute the sensitivity function. Determine it as if the data
            # weren't binned spectrally.
            scaling_fac = regrid_std_ref * ad.exposure_time() * spectral_bin
            with warnings.catch_warnings():  # division by zero
                warnings.simplefilter("ignore", category=RuntimeWarning)
                sens_func = ext.data / scaling_fac
                sens_func_var = ext.variance / scaling_fac ** 2

            # MCW 180501
            # The sensitivity function requires significant smoothing in order to
            # prevent noise from the standard being transmitted into the data
            # The easiest option is to perform a parabolic curve fit to each order
            # CJS: Remember that data have been "flatfielded" so overall response
            # of each order has been removed already!
            # import pdb; pdb.set_trace();
            sens_func_fits = []
            good = ~np.logical_or(regrid_std_ref == 0, ext.variance == 0)
            # Try to mask out deep atmospheric absorption features and unreasonable
            # measurements from low counts at the extreme orders
            #ratio = sens_func / np.percentile(sens_func, 60, axis=1)[:, np.newaxis]
            #good &= np.logical_and(ratio >= 0.2, ratio <= 5)
            #ratio = sens_func / np.nanpercentile(
            #    np.where(good, sens_func, np.nan), 60, axis=1)[:, np.newaxis]
            #good &= np.logical_and.reduce([ratio >= 0.2, ratio <= 5, ~np.isnan(ratio)])

            # CJS 240219: We fit in two setps. First has quite a harsh
            # asymmetric cut. Then points which are significantly below
            # this fit (in terms of ratio) are masked before a symmetric
            # fit is performed.
            fit_it1 = fitting.FittingWithOutlierRemoval(
                fitting.LinearLSQFitter(), sigma_clip,
                sigma_lower=1, sigma_upper=3, maxiters=2)
            fit_it2 = fitting.FittingWithOutlierRemoval(
                fitting.LinearLSQFitter(), sigma_clip,
                sigma_lower=3, sigma_upper=3, maxiters=10)
            plt.ioff()
            for od in range(sens_func.shape[0]):
                good_order = good[od]
                wavelengths = waves[od]
                min_wave, max_wave = wavelengths.min(), wavelengths.max()
                if good_order.sum() > poly_degree:
                    m_init = models.Chebyshev1D(
                        degree=poly_degree, c0=sens_func[od, good_order].mean(),
                        domain=[min_wave, max_wave])
                    m_inter, mask = fit_it1(m_init, wavelengths[good_order],
                                            sens_func[od, good_order],
                                            weights=1. / np.sqrt(sens_func_var[od, good_order]))
                    ratio = sens_func[od] / m_inter(wavelengths)
                    good_order &= ratio > min(0.9, np.median(ratio))
                    #rms = at.std_from_pixel_variations(sens_func[od, good_order])
                    #rms = (m_inter(wavelengths[good_order]) - sens_func[od, good_order])[~mask].std()
                    #good_order &= (abs(m_inter(wavelengths) - sens_func[od]) < 3 * rms)
                    #m_init = models.Chebyshev1D(
                    #    degree=poly_degree, c0=sens_func[od, good_order].mean(),
                    #    domain=[min_wave, max_wave])
                    m_final, mask2 = fit_it2(m_inter, wavelengths[good_order],
                                             sens_func[od, good_order],
                                             weights=1. / np.sqrt(sens_func_var[od, good_order]))
                    if debug_plots:
                        fig, ax = plt.subplots()
                        ax.plot(waves[od], sens_func[od], 'k-')
                        ax.plot(waves[od, good_order][~mask2],
                                sens_func[od, good_order][~mask2], 'r-')
                        ax.plot(waves[od], m_inter(waves[od]), 'g-')
                        ax.plot(waves[od], m_final(waves[od]), 'b-')
                        plt.show()
                    #rms = np.std((m_final(wavelengths) - sens_func[od])[good_order][~mask])
                    #expected_rms = np.median(np.sqrt(sens_func_var[od, good_order]))
                    #if rms > 2 * expected_rms:
                    #    log.warning(f"Unexpectedly high rms for row {od} "
                    #                f"({min_wave:.1f} - {max_wave:.1f} A)")
                    sens_func_fits.append(m_final)
                else:
                    log.warning(f"Cannot determine sensitivity for row {od} "
                                f"({min_wave:.1f} - {max_wave:.1f} A)")
                    if debug_plots:
                        fig, ax = plt.subplots()
                        ax.plot(waves[od], sens_func[od], 'k-')
                        plt.show()
                    # coefficients default to zero, so this evaluates to zero
                    sens_func_fits.append(models.Chebyshev1D(
                        degree=poly_degree, domain=[min_wave, max_wave]))
            plt.ion()

            colnames = [f"c{i}" for i in range(poly_degree+1)]
            table_data = [[getattr(m, colname).value for m in sens_func_fits]
                          for colname in colnames]
            table_data.extend([[m.domain[i] for m in sens_func_fits]
                               for i in (0, 1)])
            sensfunc_table = Table(table_data,
                                   names=colnames + ['wave_min', 'wave_max'])
            waves = make_wavelength_table(ext)
            sens_func_regrid = np.empty_like(ext.data)
            for od, sensfunc in enumerate(sens_func_fits):
                sens_func_regrid[od] = sensfunc(waves[od])

            ad[0].SENSFUNC = sensfunc_table
            ad[0].hdr['SENSFUNC'] = (sensfunc_units, "Units for SENSFUNC table")

            # Timestamp & suffix updates
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def combineOrders(self, adinputs=None, **params):
        """
        Combine the independent orders from the input ADs into one or more
        over-sampled spectra.

        The wavelength scale of the output is determined by finding the
        wavelength range of the input, and generating a new
        wavelength sampling in accordance with the ``scale`` and
        ``oversample`` parameters.

        The output spectrum is constructed as follows:

        - A blank spectrum, corresponding to the new wavelength scale, is
          initialised;
        - For each order of each input AstroData object:

            - The spectrum order is re-gridded onto the output wavelength scale;
            - The re-gridded order is averaged with the final output spectrum
              to form a new output spectrum.

          This process continues until all orders have been averaged into an
          order-combined spectrum.

        A similar process is then performed for each input AstroData object,
        with each input either producing a separate output spectrum, or being
        added to the single output either with or without calculating a scaling
        factor.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        scale : str
            Denotes what scale to generate for the final spectrum. Currently
            available are: ``'loglinear'``
        oversample : int or float
            The factor by which to (approximately) oversample the final output
            spectrum, as compared to the input spectral orders. Defaults to 1.
        stacking_mode: str
            none/None: do not stack spectra
            scaled: scale each new spectrum to the current result
            unscaled: perform no scaling
        interpolant: str
            type of resampling interpolant
        dq_threshold: float
            threshold for flagging a contaminated pixel as bad
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        scale = params["scale"]
        oversample = params["oversample"]
        stacking_mode = params["stacking_mode"] or "none"
        interpolant = params["interpolant"]
        dq_threshold = params["dq_threshold"]

        # These should probably be exposed as user parameters
        parallel = False

        adoutputs = []
        numext = set([len(ad) for ad in adinputs])
        if len(numext) != 1:
            raise ValueError("Not all inputs have the same number of extensions")
        else:
            numext = numext.pop()
        wave_limits = {arm: np.array([get_wavelength_limits(ext)
                                for ad in adinputs if ad.arm() == arm for ext in ad]).T
                       for arm in ("blue", "red")}
        ratios = {'blue': [], 'red': []}
        if scale == "loglinear":
            for ad in adinputs:
                for ext in ad:
                    waves = make_wavelength_table(ext)
                    if waves.ndim == 2:
                        ratios[ad.arm()].extend(list((waves[:, 1:] / waves[:, :-1]).ravel()))
            if not ratios:
                log.warning("Not input spectra to process as all are 1D")
                return adinputs

            wcs_models = {}
            if stacking_mode != "none":
                min_wavl = np.min([v[0].min() for v in wave_limits.values() if v.size])
                max_wavl = np.max([v[1].max() for v in wave_limits.values() if v.size])
                logspacing = np.log(np.median(np.concatenate([v for v in ratios.values()]))) / oversample
            for arm, v in ratios.items():
                if v:
                    if stacking_mode == "none":
                        min_wavl = np.min(wave_limits[arm][0])
                        max_wavl = np.max(wave_limits[arm][1])
                        logspacing = np.log(np.median(v)) / oversample
                    logmin, logmax = np.log([min_wavl, max_wavl])
                    wavl_grid = np.exp(np.linspace(
                        logmin, logmax, num=int((logmax - logmin) / logspacing)))
                    wcs_models[arm] = models.Exponential1D(
                        amplitude=wavl_grid[0],
                        tau=1. / np.log(wavl_grid[1] / wavl_grid[0]))
                    log.stdinfo(f"Resampling {arm} arm: {min_wavl:9.4f} to "
                                f"{max_wavl:9.4f}nm with {wavl_grid.size} pixels")
        else:  # can't happen yet as protected by parameters
            return adinputs

        # These arrays are only needed if we're doing some stacking
        if stacking_mode != "none":
            all_data = np.empty((numext, len(adinputs), wavl_grid.size),
                                dtype=np.float32)
            all_mask = np.empty_like(all_data, dtype=DQ.datatype)
            all_var = np.empty_like(all_data)

        # This isn't the most memory-efficient way to do this. All the input
        # ADs are order-combined first, but it frees up different algorithms
        # for combining them.
        for j, ad in enumerate(adinputs):
            wcs = wcs_models[ad.arm()]
            if j == 0 or stacking_mode == "none":
                adout = astrodata.create(ad.phu)
            if any([len(ext.shape) == 1 for ext in ad]):
                log.warning(f"No changes will be made to {ad.filename}, since "
                            f"it already contains 1D extensions")
                adoutputs.append(ad)
                continue

            for i, ext in enumerate(ad):
                waves = make_wavelength_table(ext)
                # We can easily get underflows for the VAR in np.float32
                # for flux-calibrated data
                spec_final = np.zeros(wavl_grid.size, dtype=np.float64)
                mask_final = np.zeros_like(spec_final, dtype=DQ.datatype)
                var_final = np.zeros_like(spec_final)

                # Loop over each input order, making the output spectrum the
                # result of the weighted average of itself and the order
                # spectrum
                for order in range(ext.data.shape[0]):
                    log.debug(f'Re-gridding order {order:2d}')

                    t = transform.Transform(models.Tabular1D(lookup_table=waves[order],
                                                             bounds_error=False) | wcs.inverse)
                    dg = transform.DataGroup((ext.nddata[order],), (t,),
                                             loglevel="debug")
                    dg.no_data['mask'] = DQ.no_data
                    dg.output_shape = wavl_grid.shape
                    # Note: it's never correct to conserve because data in
                    # electrons have still effectively been converted to a
                    # flux *density* by flatfielding due to different pixel
                    # wavelength extents
                    dg.transform(attributes=['data', 'mask', 'variance'],
                                 interpolant=interpolant, subsample=1,
                                 threshold=dq_threshold, conserve=False,
                                 parallel=parallel)
                    flux_for_adding = dg.output_dict['data']
                    ivar_for_adding = at.divide0(1.0, dg.output_dict['variance'].astype(np.float64))
                    ivar_for_adding[dg.output_dict['mask'] & DQ.not_signal > 0] = 0

                    spec_comp, ivar_comp = np.ma.average(
                        np.asarray([spec_final, flux_for_adding]),
                        weights=np.asarray([at.divide0(1.0, var_final),
                                            ivar_for_adding]),
                        returned=True, axis=0,
                    )
                    # Casts MaskedArrays to regular arrays
                    spec_final[:] = deepcopy(spec_comp)
                    # We want to keep the infinities
                    with np.errstate(divide="ignore"):
                        var_final[:] = 1.0 / ivar_comp.data

                # Can't use .reset without looping through extensions
                mask_final[np.logical_or.reduce([np.isnan(spec_final),
                                                 np.isinf(var_final),
                                                 var_final == 0])] = DQ.bad_pixel
                spec_final[np.isnan(spec_final)] = 0
                var_final[np.isinf(var_final)] = 0
                if stacking_mode == "none":
                    adout.append(astrodata.NDAstroData(
                        data=spec_final.astype(np.float32), mask=mask_final,
                        variance=var_final.astype(np.float32),
                        meta={'header': deepcopy(ext.hdr)}))
                else:
                    all_data[i, j] = spec_final.astype(np.float32)
                    all_mask[i, j] = mask_final
                    all_var[i, j] = var_final.astype(np.float32)

            # If we're not stacking, append this output
            if stacking_mode == "none":
                output_frame = ad[0].wcs.output_frame
                for ext in adout:
                    ext.wcs = gWCS([(adwcs.pixel_frame(1), wcs),
                                    (output_frame, None)])
                adout.hdr['DATADESC'] = ('Interpolated data',
                                         self.keyword_comments['DATADESC'])
                gt.mark_history(adout, primname=self.myself(), keyword=timestamp_key)
                adout.update_filename(suffix=params["suffix"], strip=True)
                adoutputs.append(adout)

        # We scale each input relative to the existing stack, otherwise the
        # scaling between different arms is uncertain. This way, red spectra
        # are scaled to the other red spectra (and the same for blue).
        if stacking_mode != "none":
            for i in range(numext):
                log.stdinfo(f"Combining extensions numbered {i+1}")
                for j, ad in enumerate(adinputs[1:], start=1):
                    if stacking_mode == "scaled":
                        # Combine this extension of this AD with the existing stack
                        goodpix = np.logical_and.reduce([
                            all_mask[i, j-1] == 0, all_mask[i, j] == 0,
                            all_var[i, j-1] > 0, all_var[i, j] > 0])
                        scaling = at.calculate_scaling(
                            all_data[i, j, goodpix], all_data[i, j-1, goodpix],
                            np.sqrt(all_var[i, j, goodpix]), np.sqrt(all_var[i, j-1, goodpix]))
                        log.stdinfo(f"Scaling {ad.filename} by {scaling} to match reference")
                        all_data[i, j] *= scaling
                        all_var[i, j] *= scaling * scaling
                    data, mask, var = NDStacker.wtmean(
                        data=all_data[i, j-1:j+1], mask=all_mask[i, j-1:j+1],
                        variance=all_var[i, j-1:j+1])
                    all_data[i, j] = np.nan_to_num(data)  # all bad pixels
                    all_mask[i, j] = mask
                    all_var[i, j] = var
                output = astrodata.NDAstroData(
                    data=all_data[i, -1], mask=all_mask[i, -1],
                    variance=all_var[i, -1], meta={'header': deepcopy(ext.hdr)})
                adout.append(output)
                adout[i].wcs = gWCS([(adwcs.pixel_frame(1), wcs_models[ad.arm()]),
                                     (ext.wcs.output_frame, None)])

            # Some housekeeping for data label -- as in stackFrames()
            # This will also copy the provenance of all inputs
            adout.orig_filename = adout.phu.get('ORIGNAME')
            adout.phu.set('DATALAB', f"{adout.data_label()}-STACK",
                           self.keyword_comments['DATALAB'])
            adout.hdr['DATADESC'] = ('Interpolated data',
                                     self.keyword_comments['DATADESC'])
            gt.mark_history(adout, primname=self.myself(), keyword=timestamp_key)
            adout.update_filename(suffix=params["suffix"], strip=True)
            adoutputs.append(adout)

        return adoutputs

    def darkCorrect(self, adinputs=None, **params):
        """
        Dark-correct GHOST observations.

        This primitive, at its core, simply copies the standard
        DRAGONS darkCorrect (part of :any:`Preprocess`). However, it has
        the ability to examine the binning mode of the requested dark,
        compare it to the adinput(s), and re-bin the dark to the
        correct format.

        To do this, this version of darkCorrect takes over the actual fetching
        of calibrations from :meth:`subtractDark`,
        manipulates the dark(s) as necessary,
        saves the updated dark to the present working directory, and then
        passes the updated list of dark frame(s) on to :meth:`subtractDark`.

        As a result, :any:`IOError` will be raised if the adinputs do not
        all share the same binning mode.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        dark: str/list
            name(s) of the dark file(s) to be subtracted
        do_cal: str
            controls the behaviour of this primitive
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        if params['do_cal'] == 'skip':
            log.warning("Dark correction has been turned off.")
            return adinputs

        sfx = params["suffix"]

        # Check if all the inputs have matching detector_x_bin and
        # detector_y_bin descriptors
        if not(all(
                [_.detector_x_bin() == adinputs[0].detector_x_bin() for
                 _ in adinputs])) or not(all(
            [_.detector_y_bin() == adinputs[0].detector_y_bin() for
             _ in adinputs]
        )):
            log.stdinfo('Detector x bins: %s' %
                        str([_.detector_x_bin() for _ in adinputs]))
            log.stdinfo('Detector y bins: %s' %
                        str([_.detector_y_bin() for _ in adinputs]))
            raise ValueError('Your input list of files contains a mix of '
                             'different binning modes')

        adinputs_orig = list(adinputs)
        if isinstance(params['dark'], list):
            params['dark'] = [params['dark'][i] for i in range(len(adinputs))
                              if not adinputs[i].phu.get(timestamp_key)]
        adinputs = [_ for _ in adinputs if not _.phu.get(timestamp_key)]
        if len(adinputs) != len(adinputs_orig):
            log.stdinfo('The following files have already been processed by '
                        'darkCorrect and will not be further modified: '
                        '{}'.format(', '.join([_.filename for _ in adinputs_orig
                                               if _ not in adinputs])))

        dark = params.get("dark")
        if dark is None:
            dark_list = self.caldb.get_processed_slitflat(adinputs)
        else:
            dark_list = (dark, None)

        # We need to make sure we:
        # - Provide a dark AD object for each science frame;
        # - Do not unnecessarily re-bin the same dark to the same binning
        #   multiple times
        dark_list_out = []
        dark_processing_done = {}
        for ad, dark, origin in zip(*gt.make_lists(adinputs, *dark_list,
                                           force_ad=(1,))):
            if dark is None:
                if 'qa' in self.mode:
                    log.warning(f"No changes will be made to {ad.filename}, "
                                "since no dark was specified")
                    dark_list_out.append(None)
                    continue
                else:
                    raise CalibrationNotFoundError("No processed dark listed "
                                                   f"for {ad.filename}")

            if dark.detector_x_bin() == ad.detector_x_bin() and \
                    dark.detector_y_bin() == ad.detector_y_bin():
                log.stdinfo('Binning for %s already matches input file' %
                            dark.filename)
                dark_list_out.append(dark.filename)
            else:
                xb = ad.detector_x_bin()
                yb = ad.detector_y_bin()
                dark = rebin_data(deepcopy(dark), xb, yb)
                # Re-name the dark so we don't blow away the old one on save
                dark_filename_orig = dark.filename
                dark.filename = filename_updater(dark,
                                                 suffix='_rebin%dx%d' %
                                                        (xb, yb, ),
                                                 strip=True)
                dark.write(overwrite=True)
                dark_processing_done[
                    (dark_filename_orig, xb, yb)] = dark.filename
                dark_list_out.append(dark.filename)
                log.stdinfo('Wrote out re-binned dark %s' % dark.filename)

            # Check the inputs have matching binning, and shapes
            # Copied from standard darkCorrect (primitives_preprocess)
            # TODO: Check exposure time?
            try:
                gt.check_inputs_match(ad, dark, check_filter=False)
            except ValueError:
                # Else try to extract a matching region from the dark
                log.warning('AD inputs did not match - attempting to clip dark')
                dark = gt.clip_auxiliary_data(ad, aux=dark, aux_type="cal")

                # Check again, but allow it to fail if they still don't match
                gt.check_inputs_match(ad, dark, check_filter=False)

            origin_str = f" (obtained from {origin})" if origin else ""
            log.stdinfo(f"{ad.filename}: subtracting the dark "
                        f"{dark.filename}{origin_str}")
            ad.subtract(dark)

            # Record dark used, timestamp, and update filename
            ad.phu.set('DARKIM',
                       # os.path.abspath(dark.path),
                       dark.filename,
                       self.keyword_comments["DARKIM"])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)
            add_provenance(ad, dark.filename, md5sum(dark.path) or "", self.myself())

        return adinputs_orig

    def determineWavelengthSolution(self, adinputs=None, **params):
        """
        Fit wavelength solution to a GHOST ARC frame.

        This primitive should only be applied to a reduce GHOST ARC frame. Any
        other files passed through this primitive will be skipped.

        This primitive works as follows:
        - :class:`polyfit.ghost.GhostArm` and `polyfit.extract.Extractor`
          classes are instantiated and configured for the data;
        - The ``Extractor`` class is used to find the line locations;
        - The ``GhostArm`` class is used to fit this line solution to the data.

        The primitive will use the arc line files stored in the same location
        as the initial :module:`polyfit` models kept in the ``lookups`` system.

        This primitive uses no special parameters.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        min_snr = params["min_snr"]
        sigma = params["sigma"]
        max_iters = params["max_iters"]
        radius = params["radius"]
        plot1d = params["plot1d"]
        plotrms = params["plotrms"]
        plot2d = params["debug_plot2d"]

        # import pdb; pdb.set_trace()

        # Make no attempt to check if primitive has already been run - may
        # have new calibrators we wish to apply.

        flat = params['flat']
        if flat is None:
            flat_list = self.caldb.get_processed_flat(adinputs)
        else:
            flat_list = (flat, None)

        for ad, flat, origin in zip(*gt.make_lists(adinputs, *flat_list,
                                                   force_ad=(1,))):
            if self.timestamp_keys["extractSpectra"] not in ad.phu:
                log.warning(f"extractSpectra has not been run on {ad.filename}; "
                            "skipping")
                continue

            if flat is None:
                log.warning("Could not find processed_flat calibration for "
                            f"{ad.filename}; - skipping")
                continue

            origin_str = f" (obtained from {origin})" if origin else ""
            log.stdinfo(f"{ad.filename}: using the XMOD from the "
                        f"processed_flat {flat.filename}{origin_str}")
            try:
                poly_wave = self._get_polyfit_filename(ad, 'wavemod')
                poly_spat = self._get_polyfit_filename(ad, 'spatmod')
                poly_spec = self._get_polyfit_filename(ad, 'specmod')
                poly_rot = self._get_polyfit_filename(ad, 'rotmod')
                wpars = astrodata.open(poly_wave)
                spatpars = astrodata.open(poly_spat)
                specpars = astrodata.open(poly_spec)
                rotpars = astrodata.open(poly_rot)
            except IOError:
                log.warning("Cannot open required initial model files; "
                            "skipping")
                continue

            # CJS: line_list location is now in lookups/__init__.py
            arclinefile = os.path.join(os.path.dirname(polyfit_lookup.__file__),
                                       line_list)
            arcwaves, arcfluxes = np.loadtxt(arclinefile, usecols=[1, 2]).T

            arm = GhostArm(arm=ad.arm(), mode=ad.res_mode())

            # Find locations of all significant peaks in all orders
            nm, ny = ad[0].data.shape
            all_peaks = []
            pixels = np.arange(ny)
            for m_ix, flux in enumerate(ad[0].data):
                try:
                    variance = ad[0].variance[m_ix]
                except TypeError:  # variance is None
                    nmad = median_filter(
                        abs(flux - median_filter(flux, size=2*radius+1)),
                        size=2*radius+1)
                    variance = nmad * nmad
                peaks = peak_finding.find_wavelet_peaks(
                    flux.copy(), widths=np.arange(2.5, 4.5, 0.1),
                    variance=variance, min_snr=min_snr, min_sep=5,
                    pinpoint_index=None, reject_bad=False)
                fit_g = fitting.TRFLSQFitter()  # recommended fitter
                these_peaks = []
                for x in peaks[0]:
                    good = np.zeros_like(flux, dtype=bool)
                    good[max(int(x - radius), 0):int(x + radius + 1)] = True
                    g_init = models.Gaussian1D(mean=x, amplitude=flux[good].max(),
                                               stddev=1.5)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            g = fit_g(g_init, pixels[good], flux[good])
                            if g.stddev.value < 5:  # avoid clearly bad fits
                                these_peaks.append(g)
                        except fitting.NonFiniteValueError:  # unclear why
                            pass
                all_peaks.append(these_peaks)

            # Find lines based on the extracted flux and the arc wavelengths.
            # Note that "inspect=True" also requires and input arc file, which has
            # the non-extracted data. There is also a keyword "plots".
            #lines_out = extractor.find_lines(ad[0].data, arcwaves,
            #                                 arcfile=ad[0].data,
            #                                 plots=params['plot_fit'])

            wmod_shape = wpars[0].data.shape
            m_init = WaveModel(name=ad.filename, model=wpars[0].data, arm=arm)
            arm.spectral_format_with_matrix(
                flat[0].XMOD, m_init.parameters.reshape(wmod_shape),
                spatpars[0].data, specpars[0].data, rotpars[0].data)
            extractor = Extractor(arm, None)  # slitview=None for this usage

            for iter in (0, 1):
                # Only cross-correlate on the first pass
                lines_out = extractor.match_lines(all_peaks, arcwaves,
                                                  hw=radius, xcor=not bool(iter),
                                                  log=(self.log, None)[iter])
                #lines_out is now a long vector of many parameters, including the
                #x and y position on the chip of each line, the order, the expected
                #wavelength, the measured line strength and the measured line width.
                #fitted_params, wave_and_resid = arm.read_lines_and_fit(
                #    wpars[0].data, lines_out)
                y_values, waves, orders = lines_out[:, 1], lines_out[:, 0], lines_out[:, 3]
                fit_it = fitting.FittingWithOutlierRemoval(
                    fitting.LevMarLSQFitter(), sigma_clip, sigma=sigma,
                    maxiters=max_iters)
                m_final, mask = fit_it(m_init, y_values, orders, waves)
                arm.spectral_format_with_matrix(
                    flat[0].XMOD, m_final.parameters.reshape(wmod_shape),
                    spatpars[0].data, specpars[0].data, rotpars[0].data)
                m_init = m_final

            fitted_waves = m_final(y_values, orders)
            rms = np.std((fitted_waves - waves)[~mask])
            nlines = y_values.size - mask.sum()
            log.stdinfo(f"Fit residual RMS (Angstroms): {rms:7.4f} ({nlines} lines)")
            if np.any(mask):
                log.stdinfo("The following lines were rejected:")
                for yval, order, wave, fitted, m in zip(
                        y_values, orders, waves, fitted_waves, mask):
                    if m:
                        log.stdinfo(f"    Order {int(order):2d} pixel {yval:6.1f} "
                                    f"wave {wave:10.4f} (fitted wave {fitted:10.4f})")

            if plot2d:
                plt.ioff()
                fig, ax = plt.subplots()
                xpos = lambda y, ord: arm.szx // 2 + np.interp(
                    y, pixels, arm.x_map[ord])
                for m_ix, peaks in enumerate(all_peaks):
                    for g in peaks:
                        ax.plot([g.mean.value] * 2, xpos(g.mean.value, m_ix) +
                                np.array([-30, 30]), color='darkgrey', linestyle='-')
                for (wave, yval, xval, order, amp, fwhm) in lines_out:
                    xval = xpos(yval, int(order) - arm.m_min)
                    ax.plot([yval, yval],  xval + np.array([-30, 30]), 'r-')
                    ax.text(yval + 10, xval + 30, str(wave), color='blue', fontsize=10)
                plt.show()
                plt.ion()

            if plot1d:
                plot_extracted_spectra(ad, arm, all_peaks, lines_out, mask, nrows=4)

            if plotrms:
                m_final.plotit(y_values, orders, waves, mask,
                               filename=ad.filename.replace('.fits', '_2d.pdf'))

            # CJS: Append the WFIT as an extension. It will inherit the
            # header from the science plane (including irrelevant/wrong
            # keywords like DATASEC) but that's not really a big deal.
            ad[0].WFIT = m_final.parameters.reshape(wpars[0].data.shape)

            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def extractSpectra(self, adinputs=None, **params):
        """
        Extract the object spectra from the echellogram.

        This is a primtive wrapper for a collection of :any:`polyfit <polyfit>`
        calls. For each AstroData input, this primitive:

        - Instantiates a :class:`polyfit.GhostArm` class for the input, and
          executes :meth:`polyfit.GhostArm.spectral_format_with_matrix`;
        - Instantiate :class:`polyfit.SlitView` and :class:`polyfit.Extractor`
          objects for the input
        - Extract the spectra from the input AstroData, using calls to
          :meth:`polyfit.Extractor.new_extract`.
        
        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        slit: str/None
            Name of the (processed & stacked) slit image to use for extraction
            of the profile. If not provided/set to None, the primitive will
            attempt to pull a processed slit image from the calibrations
            database (or, if specified, the --user_cal processed_slit
            command-line option)
        slitflat: str/None
            Name of the (processed) slit flat image to use for extraction
            of the profile. If not provided, set to None, the RecipeSystem
            will attempt to pull a slit flat from the calibrations system (or,
            if specified, the --user_cal processed_slitflat command-line
            option)
        flat: str/None
            Name of the (processed) flat image to use for extraction
            of the profile. If not provided, set to None, the RecipeSystem
            will attempt to pull a slit flat from the calibrations system (or,
            if specified, the --user_cal processed_flat command-line
            option)
        ifu1: str('object'|'sky'|'stowed')/None
            Denotes the status of IFU1. If None, the status is read from the
            header.
        ifu2: str('object'|'sky'|'stowed')/None
            Denotes the status of IFU2. If None, the status is read from the
            header.
        sky_subtract: bool
            subtract the sky from the object spectra?
        seeing: float/None
            can be used to create a synthetic slitviewer image (and synthetic
            slitflat image) if, for some reason, one is not available
        flat_correct: bool
            remove the reponse function from each order before extraction?
        snoise: float
            linear fraction of signal to add to noise estimate for CR flagging
        sigma: float
            number of standard deviations for identifying discrepant pixels
        weighting: str ("uniform"/"optimal")
            weighting scheme for extraction
        min_flux_frac: float (0-1)
            flag the output pixel if the unmasked flux of an object falls
            below this fractional threshold
        ftol: float
            fractional tolerance for convergence of optimal extraction flux
        apply_centroids: bool
            measure deviations from linearity of the slit profile and use
            these to apply positional offsets for lines of constant wavelength
            in the echellograms? This may simply introduce noise.
        writeResult: bool
            Denotes whether or not to write out the result of profile
            extraction to disk. This is useful for both debugging, and data
            quality assurance.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        flat_correct = params["flat_correct"]
        ifu1 = params["ifu1"]
        ifu2 = params["ifu2"]
        seeing = params["seeing"]
        snoise = params["snoise"]
        sigma = params["sigma"]
        debug_pixel = (params["debug_order"], params["debug_pixel"])
        min_flux_frac = params["min_flux_frac"]
        add_cr_map = params["debug_cr_map"]
        optimal_extraction = params["weighting"] == "optimal"
        ftol = params["ftol"]
        apply_centroids = params["apply_centroids"]
        timing = params["debug_timing"]

        # This check is to head off problems where the same flat is used for
        # multiple ADs and gets binned but then needs to be rebinned (which
        # isn't possible because the unbinned flat has been overwritten)
        binnings = set(ad.binning() for ad in adinputs)
        if len(binnings) > 1:
            raise ValueError("Not all input files have the same binning")

        # Interpret/get the necessary calibrations for all input AD objects
        slit = params["slit"]
        if slit is None:
            slit_list = self.caldb.get_processed_slit(adinputs)
        elif isinstance(slit, list):
            slit_list = [(s, None) for s in slit]
        else:
            slit_list = (slit, None)

        slitflat = params["slitflat"]
        if slitflat is None:
            slitflat_list = self.caldb.get_processed_slitflat(adinputs)
        elif isinstance(slitflat, list):
            slitflat_list = [(s, None) for s in slitflat]
        else:
            slitflat_list = (slitflat, None)

        flat = params['flat']
        if flat is None:
            flat_list = self.caldb.get_processed_flat(adinputs)
        else:
            flat_list = (flat, None)

        for (ad, slit, slit_origin, slitflat, slitflat_origin, flat,
             flat_origin) in zip(*gt.make_lists(
            adinputs, *slit_list, *slitflat_list, *flat_list,
            force_ad=(1,3,5))):
            # CJS: failure to find a suitable auxiliary file (either because
            # there's no calibration, or it's missing) places a None in the
            # list, allowing a graceful continuation.
            if flat is None:  # can't do anything as no XMOD
                raise CalibrationNotFoundError("No processed flat listed for "
                                               f"{ad.filename}")

            if slitflat is None:
                # TBD: can we use a synthetic slitflat (traceFibers in
                # makeProcessedFlat would need one too)
                raise CalibrationNotFoundError("No processed slitflat listed "
                                               f"for {ad.filename}")
                if slit is None:
                    slitflat_filename = "synthetic"
                    slitflat_data = None
                else:
                    raise CalibrationNotFoundError(
                        f"{ad.filename} has a processed slit but no processed "
                        "slitflat")
            else:
                slitflat_filename = slitflat.filename
                slitflat_data = slitflat[0].data

            if slit is None:
                if seeing is None:
                    raise CalibrationNotFoundError(
                        f"No processed slit listed for {ad.filename} and no "
                        "seeing estimate has been provided")
                else:
                    slit_filename = f"synthetic (seeing {seeing})"
                    slit_data = None
            else:
                slit_filename = slit.filename
                slit_data = slit[0].data

            # CJS: Changed to log.debug() and changed the output
            slit_origin_str = f" (obtained from {slit_origin})" if slit_origin else ""
            slitflat_origin_str = f" (obtained from {slitflat_origin})" if slitflat_origin else ""
            flat_origin_str = f" (obtained from {flat_origin})" if flat_origin else ""
            log.stdinfo(f"Slit parameters for {ad.filename}:")
            log.stdinfo(f"   processed_slit: {slit_filename}{slit_origin_str}")
            log.stdinfo(f"   processed_slitflat: {slitflat_filename}{slitflat_origin_str}")
            log.stdinfo(f"   processed_flat: {flat.filename}{flat_origin_str}")
            smoothing = flat.phu.get('SMOOTH', 0)
            if smoothing:
                log.stdinfo(f"      Flat was smoothed by {smoothing} pixels")

            res_mode = ad.res_mode()
            arm = GhostArm(arm=ad.arm(), mode=res_mode,
                           detector_x_bin=ad.detector_x_bin(),
                           detector_y_bin=ad.detector_y_bin())

            ifu_status = ["stowed", "sky", "object"]
            if ifu1 is None:
                try:
                    ifu1 = ifu_status[ad.phu['TARGET1']]
                except (KeyError, IndexError):
                    raise RuntimeError(f"{self.myself()}: ifu1 set to 'None' "
                                       f"but 'TARGET1' keyword missing/incorrect")
            if ifu2 is None:
                if res_mode == 'std':
                    try:
                        ifu2 = ifu_status[ad.phu['TARGET2']]
                    except (KeyError, IndexError):
                        raise RuntimeError(f"{self.myself()}: ifu2 set to 'None' "
                                           f"but 'TARGET1' keyword missing/incorrect")
                else:
                    ifu2 = "object" if ad.phu.get('THXE') == 1 else "stowed"

            log.stdinfo(f"\nIFU status: {ifu1} and {ifu2}")
            ifu_stowed = [obj for obj, ifu in enumerate([ifu1, ifu2])
                          if ifu == "stowed"]

            # CJS 20220411: We need to know which IFUs contain an object if
            # we need to make a synthetic slit profile so this has moved up
            if 'ARC' in ad.tags:
                # Extracts entire slit first, and then the two IFUs separately
                # v3.2.0 change: only extract individual IFUs if explicitly
                # specified by the user (since the reduction uses the full-slit)
                objs_to_use = [[]]
                requested_objects = [ifu for ifu, status in enumerate([params['ifu1'], params['ifu2']])
                                     if status == "object"]
                if requested_objects:
                    objs_to_use.append(requested_objects)
                # Since zip works off the shortest list, we can have 2-element
                # lists here even if objs_to_use has 1 element
                use_sky = [True, False]
                find_crs = [False, False]
                sky_correct_profiles = False
            else:
                ifu_selection = [obj for obj, ifu in enumerate([ifu1, ifu2])
                                 if ifu == "object"]
                objs_to_use = [ifu_selection]
                use_sky = [params["sky_subtract"] if ifu_selection else True]
                find_crs = [True]
                sky_correct_profiles = True

            # CJS: Heavy refactor. Return the filename for each calibration
            # type. Eliminates requirement that everything be updated
            # simultaneously.
            # key = self._get_polyfit_key(ad)
            # log.stdinfo("Polyfit key selected: {}".format(key))
            try:
                poly_wave = self._get_polyfit_filename(ad, 'wavemod')
                poly_spat = self._get_polyfit_filename(ad, 'spatmod')
                poly_spec = self._get_polyfit_filename(ad, 'specmod')
                poly_rot = self._get_polyfit_filename(ad, 'rotmod')
                slitv_fn = self._get_slitv_polyfit_filename(ad)
                wpars = astrodata.open(poly_wave)
                spatpars = astrodata.open(poly_spat)
                specpars = astrodata.open(poly_spec)
                rotpars = astrodata.open(poly_rot)
                slitvpars = astrodata.open(slitv_fn)
            except IOError:
                log.warning("Cannot open required initial model files for {};"
                            " skipping".format(ad.filename))
                continue

            arm.spectral_format_with_matrix(flat[0].XMOD, wpars[0].data,
                        spatpars[0].data, specpars[0].data, rotpars[0].data)
            sview_kwargs = {} if slit is None else {"binning": slit.detector_x_bin()}
            sview = SlitView(slit_data, slitflat_data,
                             slitvpars.TABLE[0], mode=res_mode,
                             microns_pix=4.54 * 180 / 50,
                             stowed=ifu_stowed, smoothing=smoothing,
                             **sview_kwargs)

            # There's no point in creating a fake slitflat first, since that
            # will case the code to use it to determine the fibre positions,
            # but the fibre positions are just the defaults
            if slit is None:
                log.stdinfo(f"Creating synthetic slit image for seeing={seeing}")
                ifus = []
                if ifu1 == 'object':
                    ifus.append('ifu0' if res_mode == 'std' else 'ifu')
                # "ifu2" is the ThXe cal fiber in HR and has no continuum
                if ifu2 == 'object' and res_mode == 'std':
                    ifus.append('ifu1')
                slit_data = sview.fake_slitimage(
                    flat_image=slitflat_data, ifus=ifus, seeing=seeing)
            elif not ad.tags.intersection({'FLAT', 'ARC'}):
                # TODO? This only models IFU0 in SR mode
                slit_models = sview.model_profile(slit_data, slitflat_data)
                log.stdinfo("")
                for k, model in slit_models.items():
                    try:
                        fwhm, apfrac = model.estimate_seeing()
                    except ValueError:
                        log.warning(f"Cannot estimate seeing in the {k} arm. "
                                    "This may be due to a lack of signal in "
                                    "the slitviewer camera that will result "
                                    "in a poor reduction. Consider using a "
                                    "synthetic slit profile model.")
                    else:
                        log.stdinfo(f"Estimated seeing in the {k} arm: {fwhm:5.3f}"
                                    f" ({apfrac*100:.1f}% aperture throughput)")

            if slitflat is None:
                log.stdinfo("Creating synthetic slitflat image")
                slitflat_data = sview.fake_slitimage()

            extractor = Extractor(arm, sview, badpixmask=ad[0].mask,
                                  vararray=ad[0].variance)
                        
            # FIXME: This really could be done as part of flat processing!
            correction = None
            if flat_correct:
                try:
                    blaze = flat[0].BLAZE
                except AttributeError:
                    log.warning(f"Flatfield {flat.filename} has no BLAZE, so"
                                "cannot perform flatfield correction")
                else:
                    ny, nx = blaze.shape
                    xbin = ad.detector_x_bin()
                    binned_blaze = blaze.reshape(ny, nx // xbin, xbin).mean(axis=-1)
                    # avoids a divide-by-zero warning
                    binned_blaze[binned_blaze < 0.0001] = np.inf
                    correction = 1. / binned_blaze

            for i, (o, s, cr) in enumerate(zip(objs_to_use, use_sky, find_crs)):
                if o:
                    log.stdinfo(f"\nExtracting objects {str(o)}; sky subtraction {str(s)}")
                elif use_sky:
                    log.stdinfo("\nExtracting entire slit")
                else:
                    raise RuntimeError("No objects for extraction and use_sky=False")

                extracted_flux, extracted_mask, extracted_var = extractor.new_extract(
                    data=ad[0].data.copy(),
                    correct_for_sky=sky_correct_profiles,
                    use_sky=s, used_objects=o, find_crs=cr,
                    snoise=snoise, sigma=sigma,
                    debug_pixel=debug_pixel,
                    correction=correction, optimal=optimal_extraction,
                    apply_centroids=apply_centroids, ftol=ftol,
                    min_flux_frac=min_flux_frac, timing=timing
                )

                # Flag pixels with VAR=0 that don't already have a flag
                extracted_mask |= (extracted_var == 0) & (extracted_mask == 0)

                # Append the extraction as a new extension... we don't
                # replace ad[0] since that will still be needed if we have
                # another iteration of the extraction loop
                nobj = extracted_flux.shape[-1]
                for obj in range(nobj):
                    ad.append(extracted_flux[:, :, obj])
                    ad[-1].mask = extracted_mask[:, :, obj]
                    ad[-1].variance = extracted_var[:, :, obj]
                    ad[-1].nddata.meta['header'] = ad[0].hdr.copy()
                    if add_cr_map and extractor.badpixmask is not None and obj == 0:
                        ad[-1].CR = extractor.badpixmask & DQ.cosmic_ray
                    if o:
                        desc_str = "sky" if s and obj == nobj-1 else f"IFU{obj+1}"
                        desc_str += f" (objects {str(o)} skysub {s})"
                    else:
                        desc_str = "entire slit"
                    ad[-1].hdr['DATADESC'] = (
                        f'Order-by-order data: {desc_str}',
                        self.keyword_comments['DATADESC'])

            del ad[0]   # Remove original echellogram data
            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)
            ad.phu.set("FLATIM", flat.filename, self.keyword_comments["FLATIM"])

            add_provenance(ad, flat.filename, md5sum(flat.path) or "", self.myself())
            if "synthetic" not in slit_filename:
                add_provenance(ad, slit_filename, md5sum(slit.path) or "", self.myself())
            if "synthetic" not in slitflat_filename:
                add_provenance(ad, slitflat_filename, md5sum(slitflat.path) or "", self.myself())

        return adinputs

    def fluxCalibrate(self, adinputs=None, ** params):
        """
        Performs flux calibration multiplying the input signal by the
        sensitivity function obtained from calculateSensitivity.

        Parameters
        ----------
        suffix :  str
            Suffix to be added to output files (default: _fluxCalibrated).
        standard: str or AstroData
            Standard star spectrum containing one extension or the same number
            of extensions as the input spectra. Each extension must have a
            `.SENSFUNC` table containing information about the overall
            sensitivity.
        units : str, optional
            Units for output spectrum (default: W m-2 nm-1).
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        std = params["standard"]
        final_units = params["units"]
        do_cal = params["do_cal"]
        sfx = params['suffix']

        if do_cal == 'skip':
            log.warning("Flux calibration has been turned off.")
            return adinputs

        # Get a suitable specphot standard (with sensitivity function)
        if std is None:
            std_list = self.caldb.get_processed_standard(adinputs)
        else:
            std_list = (std, None)

        for ad, std, origin in zip(*gt.make_lists(adinputs, *std_list,
                                    force_ad=(1,))):
            if ad.phu.get(timestamp_key):
                log.warning(f"{ad.filename}: already processed by "
                            "fluxCalibrate. Continuing.")
                continue

            if std is None:
                if 'sq' in self.mode or do_cal == 'force':
                    raise CalibrationNotFoundError("No processed standard "
                                                   f"listed for {ad.filename}")
                else:
                    log.warning(f"No changes will be made to {ad.filename}, "
                                "since no standard was specified")
                    continue

            if ad.arm() != std.arm():
                raise ValueError(f"{ad.filename} and {std.filename} are "
                                   "from different GHOST arms.")

            origin_str = f" (obtained from {origin})" if origin else ""
            log.stdinfo(f"{ad.filename}: using the standard {std.filename}"
                        f"{origin_str}")
            try:
                sensfunc = std[0].SENSFUNC
            except AttributeError:
                log.warning(f"{std.filename} has no SENSFUNC table - continuing")
                continue
            try:
                sensfunc_units = std[0].hdr['SENSFUNC']
            except KeyError:
                sensfunc_units = "W m-2 nm-1"
                log.warning("Cannot confirm units of SENSFUNC: assuming "
                            f"{sensfunc_units}")

            std_waves = make_wavelength_table(std[0])
            try:
                (1 * u.Unit(sensfunc_units)).to(
                    final_units, equivalencies=u.spectral_density(std_waves)).value
            except u.UnitConversionError:
                log.warning(f"Cannot transform units of {std.filename} - continuing")
                continue

            telescope = ad.telescope()
            exptime = ad.exposure_time()
            try:
                delta_airmass = ad.airmass() - std.airmass()
            except TypeError:  # if either airmass() returns None
                log.warning("Cannot determine airmass of target "
                            f"{ad.filename} and/or standard {std.filename}"
                            ". Not performing airmass correction.")
                delta_airmass = None
            else:
                log.stdinfo("Correcting for difference of "
                            f"{delta_airmass:5.3f} airmasses")

            spectral_bin = ad.detector_x_bin()

            for ext in ad:
                extname = f"{ad.filename} extension {ext.id}"
                if ext.hdr.get('BUNIT') != "electron":
                    log.warning(f"{extname} is not in units of electrons - "
                                "continuing")
                    continue

                sci_waves = make_wavelength_table(ext)

                if isinstance(sensfunc, Table):
                    sensfunc_regrid = np.empty_like(ext.data)
                    for od, (row, order_waves) in enumerate(zip(sensfunc, sci_waves)):
                        kwargs = {k: v for k, v in zip(sensfunc.colnames, row)}
                        m = models.Chebyshev1D(degree=len(kwargs) - 3,
                            domain=[kwargs.pop('wave_min'), kwargs.pop('wave_max')],
                                               **kwargs)
                        # Orders with no data are set to zero, but we want
                        # them to be np.inf for division
                        sens_data = m(order_waves) / spectral_bin
                        #if 358 < m(m.domain[0]) < 359:
                        #    print("TEST FIT")
                        #    x = np.arange(359, 359.2, 0.01)
                        #    print(m(x))
                        if sens_data.max() > 0:
                            sensfunc_regrid[od] = sens_data
                        else:
                            sensfunc_regrid[od] = np.inf
                    sensfunc_to_use = (sensfunc_regrid * u.Unit(sensfunc_units)).to(
                        final_units, equivalencies=u.spectral_density(sci_waves)).value
                elif ext.shape == std_waves.shape:
                        sensfunc_to_use = (sensfunc * u.Unit(sensfunc_units)).to(
                            final_units, equivalencies=u.spectral_density(sci_waves)).value
                else:
                    raise RuntimeError(
                        f"{std.filename} has a different binning to "
                        f"{ad.filename} and cannot be used as its SENSFUNC "
                        "data are stored as a data table, not a model. "
                        "You should re-reduce the standard star.")

                airmass_corr = 1.0
                if delta_airmass:
                    try:
                        extinction_correction = extinct.extinction(
                            sci_waves, telescope=telescope)
                    except KeyError:
                        log.warning(f"Telescope {telescope} not recognized. "
                                    "Not making an airmass correction.")
                    else:
                        airmass_corr = 10 ** (0.4 * delta_airmass * extinction_correction)

                scaling_factor = exptime * sensfunc_to_use / airmass_corr
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    ext.data /= scaling_factor
                    # has overflows if we square scaling_factor first!
                    ext.variance /= scaling_factor
                    ext.variance /= scaling_factor

            # Make the relevant header update
            ad.hdr['BUNIT'] = final_units

            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)
            add_provenance(ad, std.filename, md5sum(std.path) or "", self.myself())

        return adinputs

    def makeIRAFCompatible(self, adinputs=None, suffix=None):
        """
        Write out the extracted spectra in a format that is compatible with IRAF.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        for ad in adinputs:
            updated = False
            hdulist = astrodata.fits.ad_to_hdulist(ad)
            for hdu in hdulist:
                if hdu.data is not None and len(hdu.data.shape) == 1:
                    hdr = hdu.header
                    if (hdr.get('DC-FLAG') != 1 and
                            hdr.get('CTYPE1', '').endswith("LOG")):
                        cdelt1 = hdr['CD1_1']
                        crval1 = hdr['CRVAL1']
                        crpix1 = hdr['CRPIX1']
                        x = np.arange(hdu.data.size)
                        w = crval1 * np.exp(cdelt1 * (x + 1 - crpix1) / crval1)
                        dw = np.log10(w[1] / w[0])
                        new_kws = {"CRPIX1": crpix1, "CRVAL1": np.log10(crval1),
                                   "CDELT1": dw, "CD1_1": dw, "DC-FLAG": 1}
                        hdr.update(new_kws)
                        updated = True
            if updated:
                new_filename = filename_updater(ad, suffix=suffix)
                hdulist.writeto(new_filename, overwrite=True)
                log.stdinfo(f"{ad.filename} updated and written to disk "
                            f"as {new_filename}")
            else:
                log.stdinfo(f"No changes were made to the header of {ad.filename}")

        return adinputs

    def measureBlaze(self, adinputs=None, **params):
        """
        This primitive measures the blaze function in each order by summing
        the signal at each wavelength pixel in a processed_flat and
        normalizing. The result is then appended as a 2D array with the
        extension name "BLAZE".

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        slitflat: str or :class:`astrodata.AstroData` or None
            slit flat to use; if None, the calibration system is invoked
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params['suffix']
        order = params.get('order')
        sigma = 3
        max_iters = 5

        slitflat = params["slitflat"]
        if slitflat is None:
            flat_list = self.caldb.get_processed_slitflat(adinputs)
        else:
            flat_list = (slitflat, None)

        for ad, slitflat, origin in zip(*gt.make_lists(adinputs, *flat_list,
                                                       force_ad=(1,))):
            if slitflat is None:
                raise CalibrationNotFoundError("No processed_slitflat found "
                                               f"for {ad.filename}")

            res_mode = ad.res_mode()
            arm = GhostArm(arm=ad.arm(), mode=res_mode,
                           detector_x_bin=ad.detector_x_bin(),
                           detector_y_bin=ad.detector_y_bin())

            origin_str = f" (obtained from {origin})" if origin else ""
            log.stdinfo(f"{ad.filename}: using slit profile from the "
                        f"processed_slitflat {slitflat.filename}{origin_str}")
            try:
                poly_wave = self._get_polyfit_filename(ad, 'wavemod')
                poly_spat = self._get_polyfit_filename(ad, 'spatmod')
                poly_spec = self._get_polyfit_filename(ad, 'specmod')
                poly_rot = self._get_polyfit_filename(ad, 'rotmod')
                slitv_fn = self._get_slitv_polyfit_filename(ad)
                wpars = astrodata.open(poly_wave)
                spatpars = astrodata.open(poly_spat)
                specpars = astrodata.open(poly_spec)
                rotpars = astrodata.open(poly_rot)
                slitvpars = astrodata.open(slitv_fn)
            except IOError:
                raise RuntimeError("Cannot open required initial model files; "
                                   "skipping")

            smoothing = ad.phu.get('SMOOTH', 0)
            arm.spectral_format_with_matrix(ad[0].XMOD, wpars[0].data,
                        spatpars[0].data, specpars[0].data, rotpars[0].data)
            sview = SlitView(slitflat[0].data, slitflat[0].data,
                             slitvpars.TABLE[0], mode=res_mode,
                             microns_pix=4.54 * 180 / 50,
                             binning=slitflat.detector_x_bin(),
                             smoothing=smoothing)
            extractor = Extractor(arm, sview, badpixmask=ad[0].mask,
                                  vararray=ad[0].variance)
            extracted_flux, extracted_var, extracted_mask = extractor.quick_extract(ad[0].data)
            med_flux = np.median(extracted_flux[extracted_mask == 0])
            extracted_flux /= med_flux

            # Due to poorly-determined gains, we can't fit a function and use
            # that: there's a jump at the middle of each order. But I'll leave
            # this code here, in case that changes.
            if order is not None:
                for i, (flux, var, mask) in enumerate(
                        zip(extracted_flux, extracted_var, extracted_mask)):
                    ny = extracted_flux.shape[1]
                    m_init = models.Chebyshev1D(degree=order, domain=[0, ny-1])
                    fit_it = fitting.FittingWithOutlierRemoval(
                        fitting.LinearLSQFitter(), sigma_clip, sigma=sigma,
                        maxiters=max_iters)
                    m_final, _ = fit_it(m_init, np.arange(ny),
                                        np.ma.masked_where(mask, flux),
                                        weights=at.divide0(1., np.sqrt(var)))
                    print(f"Plotting {arm.m_min+i}")
                    fig, ax = plt.subplots()
                    ally = np.arange(ny)
                    ax.plot(ally[mask==0], flux[mask==0], 'bo')
                    ax.plot(ally[_], flux[_], 'ro')
                    ax.plot(ally, m_final(ally), 'k-')
                    plt.show()

            ad[0].BLAZE = extracted_flux
            ad[0].BLAZE[extracted_mask.astype(bool)] = 0
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)
            add_provenance(ad, slitflat.filename, md5sum(slitflat.path) or "", self.myself())

        return adinputs

    def removeScatteredLight(self, adinputs=None, **params):
        """
        This primitive attempts to remove the scattered light in the GHOST
        echellograms, by measuring the signal at different locations between
        the echelle orders and fitting a smooth surface to them. It uses the
        XMOD (from an associated processed_flat if the input image is not a
        flat itself) to determine the locations of the orders and ignores
        pixels in the input image's mask. The signal is determined from a
        percentile which is normally less than 50 in order to avoid being

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        skip: bool
            skip primitive entirely?
        xsampling: int
            sampling of inter-order light in x (wavelength) direction (unbinned pixels)
        debug_spline_smoothness: float
            scaling factor for spline smoothness
        debug_percentile: float
            percentile for obtaining inter-order light level
        debu_avoidance: int
            number of (unbinned)
        debug_save_model: bool
            attach scattered light model to output
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        if params["skip"]:
            log.stdinfo("Not removing scattered light since skip=True")
            return adinputs

        xsampling = params["xsampling"]
        smoothness = params["debug_spline_smoothness"]
        percentile = params["debug_percentile"]
        avoidance = params["debug_avoidance"]
        save_model = params["debug_save_model"]
        flat_list = self.caldb.get_processed_flat(adinputs)

        warn_msg = (f"{self.myself()} has not been fully tested. "
                    "It is strongly recommended that you ")
        if not save_model:
            warn_msg += "run with debug_save_model=True and "
        warn_msg += "review the scattered light model."
        log.warning(warn_msg)

        for ad, flat, origin in zip(*gt.make_lists(adinputs, *flat_list, force_ad=(1,))):
            if len(ad) > 1:
                log.warning(f"{ad.filename} has more than one extension - "
                            "ignoring")
                continue
            if flat is None:
                if 'FLAT' in ad.tags:
                    flat = ad
                    origin = "the input"
                else:
                    log.warning(f"{ad.filename}: no flat found - "
                                "cannot correct for scattered light")
                    continue
            res_mode = ad.res_mode()
            xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()
            arm = GhostArm(arm=ad.arm(), mode=res_mode,
                           detector_x_bin=xbin,
                           detector_y_bin=ybin)
            try:
                poly_wave = self._get_polyfit_filename(ad, 'wavemod')
                poly_spat = self._get_polyfit_filename(ad, 'spatmod')
                poly_spec = self._get_polyfit_filename(ad, 'specmod')
                poly_rot = self._get_polyfit_filename(ad, 'rotmod')
                slitv_fn = self._get_slitv_polyfit_filename(ad)
                wpars = astrodata.open(poly_wave)
                spatpars = astrodata.open(poly_spat)
                specpars = astrodata.open(poly_spec)
                rotpars = astrodata.open(poly_rot)
                slitvpars = astrodata.open(slitv_fn)
            except IOError:
                raise RuntimeError("Cannot open required initial model files; "
                                   "skipping")

            origin_str = f" (obtained from {origin})" if origin else ""
            log.stdinfo(f"{ad.filename}: using XMOD from the "
                        f"processed_flat {flat.filename}{origin_str}")
            arm.spectral_format_with_matrix(flat[0].XMOD, wpars[0].data,
                        spatpars[0].data, specpars[0].data, rotpars[0].data)
            # We don't need a slit image because everything is handled by
            # slitvpars determining the length of the slit
            sview = SlitView(np.ones((200, 200)), None,
                             slitvpars.TABLE[0], mode=res_mode)
            extractor = Extractor(arm, sview, badpixmask=ad[0].mask,
                                  vararray=ad[0].variance)
            unilluminated = np.ones_like(ad[0].data, dtype=bool)

            profiles = [extractor.slitview.slit_profile(arm=ad.arm())]
            n_slitpix = profiles[0].size
            profile_y_microns = (np.arange(n_slitpix) -
                                 n_slitpix / 2 + 0.5) * sview.microns_pix
            try:
                x_map, w_map, blaze, matrices = extractor.bin_models()
            except Exception:
                raise RuntimeError('Extraction failed, unable to bin models.')
            print("    Modeling order ", end="")
            nm, ny = x_map.shape
            nx = int(arm.szx / arm.xbin)

            import sys
            from .polyfit.extract import resample_slit_profiles_to_detector

            for i in range(nm):
                print(f"{arm.m_min + i}...", end="")
                sys.stdout.flush()

                for j in range(ny):
                    x_ix, phi, profiles = resample_slit_profiles_to_detector(
                        profiles, profile_y_microns, x_map[i, j] + nx // 2,
                        detpix_microns=matrices[i, j, 0, 0])
                    if avoidance >= ybin:
                        x_ix = np.r_[x_ix, np.arange(x_ix.min() - avoidance // ybin, x_ix.min()),
                                     np.arange(x_ix.max()+1, x_ix.max() + avoidance // ybin)]
                        x_ix = np.minimum(np.maximum(x_ix, 0), nx-1)
                    _slice = (j, x_ix) if extractor.transpose else (x_ix, j)
                    unilluminated[_slice] = False
            print("\n")

            # Mark all pixels outside the topmost/bottommost orders as
            # illuminated (i.e., do not use in the fit)
            if arm.arm == "red":
                for ix, iy in enumerate(unilluminated.argmin(axis=0)):
                    unilluminated[iy - 256 // ybin:iy, ix] = False
            for ix, iy in enumerate(unilluminated[::-1].argmin(axis=0)):
                if iy > 0:
                    unilluminated[-iy:, ix] = False
            #        #if arm.arm == "red":
            #        #    unilluminated[-iy-768//ybin:, ix] = False

            ny, nx = ad[0].shape

            interp_points = []
            xs = xsampling // xbin
            y = np.repeat(np.arange(ny)[:, np.newaxis], xs, axis=1)
            regions, nregions = measurements.label(unilluminated)
            if ad[0].mask is not None:
                regions[ad[0].mask > 0] = 0
            for ix in range(0, nx, xs):
                _slice = (slice(None), slice(ix, ix+xs))
                for i in range(1, nregions+1):
                    points = (regions[_slice] == i)
                    if points.any():
                        interp_points.append(
                            [ix + 0.5 * (xs-1), np.mean(y[points]),
                             np.percentile(ad[0].data[_slice][points],
                                           percentile)])
            interp_points = np.asarray(interp_points).T

            spline = interpolate.SmoothBivariateSpline(
                *interp_points, bbox=[0, nx-1, 0, ny-1],
                s=smoothness*interp_points.shape[1])
            scattered_light = spline(np.arange(nx), np.arange(ny), grid=True).T
            scattered_light = np.maximum(scattered_light, 0)

            if save_model:
                ad_scatt = astrodata.create(ad.phu)
                ad_scatt.append(scattered_light)
                ad_scatt.update_filename(suffix="_scatteredLightModel", strip=True)
                log.stdinfo(f"Saving scattered light model as {ad_scatt.filename}")
                ad_scatt.write(overwrite=True)
            ad[0].subtract(scattered_light)

            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)
        return adinputs

    def scaleCountsToReference(self, adinputs=None, **params):
        """
        This primitive scales the input images so that the scaled fluxes of
        the sources in the OBJCAT match those in the reference image (the
        first image in the list). By setting the input parameter tolerance=0,
        it is possible to simply scale the images by the exposure times.
        The scaling is done via a pixel-by-pixel comparison, assuming that
        the wavelength scales are sufficiently close (a check is made for
        this), but it could be altered to interpolate each input to the
        reference.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        tolerance: float (0 <= tolerance <= 1)
            tolerance within which scaling must match exposure time to be used
        use_common: bool
            use only sources common to all frames?
        radius: float
            matching radius in arcseconds
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        tolerance = params["tolerance"]

        if len(adinputs) <= 1:
            log.stdinfo("No scaling will be performed, since at least two "
                        f"AstroData objects are required for {self.myself()}")
            return adinputs

        shapes = set([ext.shape for ad in adinputs for ext in ad])
        if len(shapes) > 1:
            raise ValueError("Not all inputs have the same shape")

        ext_ids = [[i for i, datadesc in enumerate(ad.hdr.get('DATADESC', ''))
                    if 'IFU' in datadesc] for ad in adinputs]
        if ext_ids != ext_ids[::-1]:
            raise ValueError("Not all inputs have the same target extensions")
        ext_ids = ext_ids.pop()
        log.stdinfo(f"Objects found in extensions {[i+1 for i in ext_ids]}")
        data = np.empty((4, len(ext_ids)) + shapes.pop(), dtype=np.float32)

        ref_ad = adinputs[0]
        kw_exptime = ref_ad._keyword_for('exposure_time')
        ref_texp = ref_ad.exposure_time()
        scale_factors = [1]  # for first (reference) image
        exptimes = [ref_texp]
        ref_wave = make_wavelength_table(ref_ad[0])

        for j, ad in enumerate(adinputs[1:], start=1):
            if np.max(abs(make_wavelength_table(ad[0]) - ref_wave)) > 0.1:
                log.warning(f"Wavelength axes of {ref_ad.filename} and "
                            f"{ad.filename} differ by more than 0.1nm")
            texp = ad.exposure_time()
            exptimes.append(texp)
            time_scaling = ref_texp / texp
            if tolerance == 0:
                scale_factors.append(time_scaling)
                continue
            data[:2] = [[adinputs[jj][i].data for i in ext_ids] for jj in (j, 0)]
            data[2:] = [[np.sqrt(adinputs[jj][i].variance) for i in ext_ids] for jj in (j, 0)]
            goodpix = np.asarray([np.logical_and.reduce([
                ref_ad[i].mask == 0, ad[i].mask == 0,
                ref_ad[i].variance > 0, ad[i].variance > 0]) for i in ext_ids])
            scaling = at.calculate_scaling(
                x=data[0, goodpix], y=data[1, goodpix],
                sigma_x=np.sqrt(data[2, goodpix]), sigma_y=np.sqrt(data[3, goodpix]))
            if (scaling > 0 and (tolerance == 1 or
                                 ((1 - tolerance) <= scaling <= 1 / (1 - tolerance)))):
                scale_factors.append(scaling)
            else:
                log.warning(f"Scaling factor {scaling:.3f} for {ad.filename} "
                            f"is inconsisent with exposure "
                            f"time scaling {time_scaling:.3f}")
                scale_factors.append(time_scaling)

        for ad, scaling, exptime in zip(adinputs, scale_factors, exptimes):
            log.stdinfo(f"Scaling {ad.filename} by {scaling:.3f}")
            if scaling != 1:
                ad.multiply(scaling)
                # ORIGTEXP should always be the *original* exposure
                # time, so if it already exists, leave it alone!
                if "ORIGTEXP" not in ad.phu:
                    ad.phu.set("ORIGTEXP", exptime, "Original exposure time")
                # The new exposure time should probably be the reference's
                # exposure time, so that all the outputs have the same value
                ad.phu.set(kw_exptime, ref_texp,
                           comment=self.keyword_comments[kw_exptime])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)

        return adinputs

    def stackArcs(self, adinputs=None, **params):
        """
        This primitive stacks input arc frames by associating arcs taken in
        close temporal proximity, and stacking them together.

        This primitive works by 'clustering' the input arcs, and then calling
        the standard stackFrames primitive on each cluster in turn.

        Parameters
        ----------
        time_delta : float
            The time delta between arcs that will allow for association,
            expressed in minutes. Note that this time_delta is between
            two arcs in sequence; e.g., if time_delta is 20 minutes, and arcs
            are taken with the following gaps:
            A <- 19m -> B <- 10m -> C <- 30m -> D <- 19m -> E
            These arcs will be clustered as follows:
            [A, B, C] and [D, E]
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        time_delta = params['time_delta']
        stack_params = self._inherit_params(params, "stackFrames")

        # import pdb; pdb.set_trace()

        processed = [ad.filename for ad in adinputs if timestamp_key in ad.phu]
        if processed:
            raise RuntimeError("The following frames have already been "
                               "processed by stackArcs\n    " +
                               "\n    ".join(processed))

        clusters = []
        if time_delta is None:
            parent_names = [ad.phu['ORIGNAME'].split('.')[0][:-3]
                            for ad in adinputs]
            # Cluster by bundle (i.e. ORIGNAME')
            for bundle in set(parent_names):
                clusters.append([ad for ad, parent in zip(adinputs, parent_names)
                                 if parent == bundle])
        else:
            # Sort the input arc files by DATE-OBS/UTSTART
            adinputs.sort(key=lambda x: x.ut_datetime())

            # Cluster the inputs
            for ad in adinputs:
                try:
                    ref_time = clusters[-1][-1].ut_datetime()
                except IndexError:
                    # Start a new cluster
                    clusters.append([ad, ])
                    continue

                if (np.abs(ad.ut_datetime() - ref_time).total_seconds()
                           < time_delta):
                    # Append to the last cluster
                    clusters[-1].append(ad)
                else:
                    # Start a new cluster
                    clusters.append([ad, ])

        # Stack each cluster
        for i, cluster in enumerate(clusters):
            if len(cluster) == 0:
                raise RuntimeError("I've ended up with an empty cluster...")

            # Don't need to touch a cluster with only 1 AD
            if len(cluster) > 1:
                clusters[i] = self.stackFrames(adinputs=cluster, **stack_params)

        # Flatten out the list
        clusters_flat = [item for sublist in clusters for item in sublist]

        # Book keeping
        for i, ad in enumerate(clusters_flat):
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)

        return clusters_flat

    def standardizeInstrumentHeaders(self, adinputs=None, **params):
        """
        This primitive updates the SATURATE keyword in the headers, because
        it is (erroneously) set to 0 for the blue spectrograph
        """
        kw = adinputs[0]._keyword_for('saturation_level')
        for ad in adinputs:
            for ext, sat_level in zip(ad, ad.saturation_level()):
                ext.hdr[kw] = sat_level
        return adinputs

    def standardizeStructure(self, adinputs=None, **params):
        """
        The Gemini-level version of this primitive
        will try to attach an MDF because a GHOST image is
        tagged as SPECT. Rather than set parameters for that primitive to
        stop it from doing so, just override with a no-op primitive.
        
        .. note::
            This could go in primitives_ghost.py if the SLITV version
            also no-ops.
        """
        return adinputs

    def standardizeSpectralFormat(self, adinputs=None, suffix=None):
        """
        Convert the input file into multiple apertures, each with its own
        gWCS object to describe the wavelength scale. This allows the
        spectrum (or each order) to be displayed by the "dgsplot" script

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        input_frame = cf.CoordinateFrame(naxes=1, axes_type=['SPATIAL'],
                                         axes_order=(0,), name="pixels",
                                         axes_names=['x'], unit=u.pix)
        output_frame = cf.SpectralFrame(axes_order=(0,), unit=u.nm,
                                        axes_names=["WAVE"],
                                        name="Wavelength in vacuo")

        wave_tol = 1e-4  # rms limit for a linear/loglinear wave model
        adoutputs = []
        for ad in adinputs:
            adout = astrodata.create(ad.phu)
            adout.update_filename(suffix=suffix, strip=True)
            log.stdinfo(f"Converting {ad.filename} to {adout.filename}")
            for i, ext in enumerate(ad, start=1):
                if not hasattr(ext, "WAVL"):
                    log.warning(f"    EXTVER {i} has no WAVL table. Ignoring.")
                    continue
                npix, nobj = ext.shape[-2:]
                try:
                    orders = list(range(ext.shape[-3]))
                except:
                    orders = [None]
                wave_models = []
                for order in orders:
                    wave = 0.1 * ext.WAVL[order].ravel()  # because WAVL is always 2D
                    linear = np.diff(wave).std() < wave_tol
                    loglinear = (wave[1:] / wave[:-1]).std() < wave_tol
                    if linear or loglinear:
                        if linear:
                            log.debug(f"Linear wavelength solution found for order {order}")
                            fit_it = fitting.LinearLSQFitter()
                            m_init = models.Polynomial1D(
                                degree=1, c0=wave[0], c1=wave[1]-wave[0])
                        else:
                            log.debug(f"Loglinear wavelength solution found for order {order}")
                            fit_it = fitting.LevMarLSQFitter()
                            m_init = models.Exponential1D(
                                amplitude=wave[0], tau=1./np.log(wave[1] / wave[0]))
                        wave_model = fit_it(m_init, np.arange(npix), wave)
                        log.debug("    "+" ".join([f"{p}: {v}" for p, v in zip(
                            wave_model.param_names, wave_model.parameters)]))
                    else:
                        log.debug(f"Using tabular model for order {order}")
                        wave_model = models.Tabular1D(np.arange(npix), wave)
                    wave_model.name = "WAVE"
                    wave_models.append(wave_model)

                for spec in range(nobj):
                    for order, wave_model in zip(orders, wave_models):
                        ndd = ext.nddata.__class__(data=ext.data[order, :, spec].ravel(),
                                                   meta={'header': ext.hdr.copy()})
                        if ext.has_mask():
                            ndd.mask = ext.mask[order, :, spec].ravel()
                        if ext.has_variance():
                            ndd.variance = ext.variance[order, :, spec].ravel()
                        adout.append(ndd)
                        adout[-1].hdr[ad._keyword_for('data_section')] = f"[1:{npix}]"
                        adout[-1].wcs = gWCS([(input_frame, wave_model),
                                              (output_frame, None)])

                log.stdinfo(f"    EXTVER {i} has been converted to EXTVERs "
                            f"{len(adout) - nobj * len(orders) + 1}-{len(adout)}")

            adoutputs.append(adout)

        return adoutputs

    def traceFibers(self, adinputs=None, **params):
        """
        Locate the fiber traces, parametrized by an :any:`polyfit` model.

        The primitive locates the slit apertures within a GHOST frame,
        and inserts a :any:`polyfit` model into a new extension on each data
        frame. This model is placed into a new ``.XMOD`` attribute on the
        extension.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        slitflat: str or :class:`astrodata.AstroData` or None
            slit flat to use; if None, the calibration system is invoked
        smoothing: int
            Gaussian FWHM (in unbinned slitviewer pixels) for smoothing the
            slit profile
        make_pixel_model: bool
            add a PIXMODEL extension with a model of the fiber traces?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        make_pixel_model = params.get('make_pixel_model', False)
        smoothing = params["smoothing"]

        # Make no attempt to check if primitive has already been run - may
        # have new calibrators we wish to apply.

        # CJS: See comment in extractSpectra() for handling of calibrations
        slitflat = params["slitflat"]
        if slitflat is None:
            flat_list = self.caldb.get_processed_slitflat(adinputs)
        else:
            flat_list = (slitflat, None)

        for ad, slit_flat, origin in zip(*gt.make_lists(adinputs, *flat_list,
                                                        force_ad=(1,))):
            if not {'PREPARED', 'GHOST', 'FLAT'}.issubset(ad.tags):
                log.warning(f"{self.myself()} is only run on prepared flats: "
                            f"{ad.filename} will not be processed")
                continue

            if slit_flat is None:
                raise CalibrationNotFoundError("No processed_slitflat found "
                                               f"for {ad.filename}")

            origin_str = f" (obtained from {origin})" if origin else ""
            log.stdinfo(f"{ad.filename}: using slitflat "
                        f"{slit_flat.filename}{origin_str}")
            try:
                poly_xmod = self._get_polyfit_filename(ad, 'xmod')
                log.stdinfo(f'Found xmod: {poly_xmod}')
                poly_spat = self._get_polyfit_filename(ad, 'spatmod')
                log.stdinfo(f'Found spatmod: {poly_spat}')
                slitv_fn = self._get_slitv_polyfit_filename(ad)
                log.stdinfo(f'Found slitvmod: {slitv_fn}')
                xpars = astrodata.open(poly_xmod)
                spatpars = astrodata.open(poly_spat)
                slitvpars = astrodata.open(slitv_fn)
            except IOError:
                log.warning("Cannot open required initial model files; "
                            "skipping")
                continue

            arm = ad.arm()
            res_mode = ad.res_mode()
            ghost_arm = GhostArm(arm=arm, mode=res_mode)

            # Create an initial model of the spectrograph
            xx, wave, blaze = ghost_arm.spectral_format(xparams=xpars[0].data)

            slitview = SlitView(slit_flat[0].data, slit_flat[0].data,
                                slitvpars.TABLE[0], mode=res_mode,
                                microns_pix=4.54 * 180 / 50,
                                binning=slit_flat.detector_x_bin(),
                                smoothing=smoothing)

            # Convolve the flat field with the slit profile
            flat_conv = ghost_arm.slit_flat_convolve(
                np.ma.masked_array(ad[0].data, mask=ad[0].mask),
                slit_profile=slitview.slit_profile(arm=arm),
                spatpars=spatpars[0].data, microns_pix=slitview.microns_pix,
                xpars=xpars[0].data
            )

            # Fit the initial model to the data being considered
            fitted_params = ghost_arm.fit_x_to_image(flat_conv,
                                                     xparams=xpars[0].data,
                                                     decrease_dim=8,
                                                     sampling=2,
                                                     inspect=False)

            # CJS: Append the XMOD as an extension. It will inherit the
            # header from the science plane (including irrelevant/wrong
            # keywords like DATASEC) but that's not really a big deal.
            # (The header can be modified/started afresh if needed.)
            ad[0].XMOD = fitted_params

            # MJI: Compute a pixel-by-pixel model of the flat field from the new XMOD and
            # the slit image.
            if make_pixel_model:
                try:
                    poly_wave = self._get_polyfit_filename(ad, 'wavemod')
                    poly_spec = self._get_polyfit_filename(ad, 'specmod')
                    poly_rot = self._get_polyfit_filename(ad, 'rotmod')
                    wpars = astrodata.open(poly_wave)
                    specpars = astrodata.open(poly_spec)
                    rotpars = astrodata.open(poly_rot)
                except IOError:
                    log.warning("Cannot open required initial model files "
                                "for PIXELMODEL; skipping")
                    continue

                # Create an extractor instance, so that we can add the pixel model to the
                # data.
                ghost_arm.spectral_format_with_matrix(ad[0].XMOD, wpars[0].data,
                                                      spatpars[0].data, specpars[0].data, rotpars[0].data)
                extractor = Extractor(ghost_arm, slitview, badpixmask=ad[0].mask,
                                      vararray=ad[0].variance)
                pixel_model = extractor.make_pixel_model()
                ad[0].PIXELMODEL = pixel_model

            if smoothing:
                ad.phu['SMOOTH'] = (smoothing, "Pixel FWHM of SVC smoothing kernel")
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)
            add_provenance(ad, slit_flat.filename, md5sum(slit_flat.path) or "", self.myself())

        return adinputs

    def write1DSpectra(self, adinputs=None, **params):
        """
        Write 1D spectra to files listing the wavelength and data (and
        optionally variance and mask) in one of a range of possible formats.

        Because Spect.write1DSpectra requires APERTURE numbers, the GHOST
        version of this primitive adds them before calling the Spect version.

        Parameters
        ----------
        format : str
            format for writing output files
        header : bool
            write FITS header before data values?
        extension : str
            extension to be used in output filenames
        apertures : str
            comma-separated list of aperture numbers to write
        dq : bool
            write DQ (mask) plane?
        var : bool
            write VAR (variance) plane?
        overwrite : bool
            overwrite existing files?
        wave_units: str
            units of the x (wavelength/frequency) column
        data_units: str
            units of the data column
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        for ad in adinputs:
            for i, ext in enumerate(ad, start=1):
                ext.hdr['APERTURE'] = i
        Spect.write1DSpectra(self, adinputs, **params)
        return adinputs

    @staticmethod
    def _has_valid_extensions(ad):
        return len(ad) == 4


##############################################################################
# Below are the helper functions for the user level functions in this module #
##############################################################################

    def _get_polyfit_filename(self, ad, caltype):
        """
        Gets the filename of the relevant initial polyfit file for this
        input GHOST science image

        This primitive uses the arm, resolution mode and observing epoch
        of the input AstroData object to determine the correct initial
        polyfit model to provide. The model provided matches the arm and
        resolution mode of the data, and is the most recent model generated
        before the observing epoch.

        Parameters
        ----------
        ad : :class:`astrodata.AstroData`
            AstroData object to return the relevant initial model filename for
        caltype : str
            The initial model type (e.g. ``'rotmod'``, ``'spatmod'``, etc.)
            requested. An :any:`AttributeError` will be raised if the requested
            model type does not exist.

        Returns
        -------
        str/None:
            Filename (including path) of the required polyfit file
        """
        return polyfit_lookup.get_polyfit_filename(self.log, ad.arm(),
                                                   ad.res_mode(), ad.ut_date(),
                                                   ad.filename, caltype)

    def _get_slitv_polyfit_filename(self, ad):
        return polyfit_lookup.get_polyfit_filename(self.log, 'slitv',
                                                   ad.res_mode(), ad.ut_date(),
                                                   ad.filename, 'slitvmod')

    def _request_bracket_arc(self, ad, before=None):
        """
        Request the 'before' or 'after' arc for the passed ad object.

        For maximum accuracy in wavelength calibration, GHOST data is calibrated
        the two arcs taken immediately before and after the exposure. However,
        the Gemini calibration system is not rigged to perform such logic (it
        can return multipled arcs, but cannot guarantee that they straddle
        the observation in time).

        This helper function works by doing the following:

        - Append a special header keyword, 'ARCBEFOR', to the PHU. This keyword
          will be True if a before arc is requested, or False if an after arc
          is wanted.
        - getProcessedArc is the invoked, followed by the _get_cal call. The
          arc calibration association rules will see the special descriptor
          related to the 'ARCBEFOR' header keyword, and fetch an arc
          accordingly.
        - The header keyword is then deleted from the ad object, returning it
          to its original state.

        Parameters
        ----------
        before : bool
            Denotes whether to ask for the most recent arc before (True) or
            after (False) the input AD was taken. Defaults to None, at which
            point :any:`ValueError` will be thrown.

        Returns
        -------
        arc_ad : astrodata.AstroData instance (or None)
            The requested arc. Will return None if no suitable arc is found.
        """
        if before is None:
            raise ValueError('_request_bracket_arc requires that the before '
                             'kwarg is either True or False. If you wish to '
                             'do a "standard" arc calibration fetch, simply '
                             'use getProcessedArc directly.')

        ad.phu['ARCBEFOR'] = before
        arc_ad = self.caldb.get_processed_arc([ad]).items()[0][0]
        del ad.phu['ARCBEFOR']
        # If the arc is retrieved from user_cals then it will ignore 'ARCBEFOR'
        # and the same file will be returned twice, but we don't want that
        if arc_ad:
            arc_ad = astrodata.open(arc_ad)
            correct_timing = before == (arc_ad.ut_datetime() < ad.ut_datetime())
            return arc_ad if correct_timing else None
        return None


def plot_extracted_spectra(ad, arm, all_peaks, lines_out, mask=None, nrows=4):
    """
    Produce plot of all the extracted orders, located peaks, and matched
    arc lines. Abstracted here to make the primitive cleaner.
    """
    if mask is None:
        mask = np.zeros(len(lines_out), dtype=bool)
    pixels = np.arange(arm.szy)
    with PdfPages(ad.filename.replace('.fits', '.pdf')) as pdf:
        for m_ix, (flux, peaks) in enumerate(zip(ad[0].data, all_peaks)):
            order = m_ix + arm.m_min
            if m_ix % nrows == 0:
                if m_ix > 0:
                    fig.subplots_adjust(hspace=0)
                    pdf.savefig(bbox_inches='tight')
                fig, axes = plt.subplots(nrows, 1, sharex=True)
            ax = axes[m_ix % nrows]
            ax.plot(pixels, flux, color='darkgrey', linestyle='-', linewidth=1)
            ymax = flux.max()
            for g in peaks:
                x = g.mean.value + np.arange(-3, 3.01, 0.1) * g.stddev.value
                ax.plot(x, g(x), 'k-', linewidth=1)
                ymax = max(ymax, g.amplitude.value)
            for (wave, yval, xval, ord, amp, fwhm), m in zip(lines_out, mask):
                if ord == order:
                    kwargs = {"fontsize": 5, "rotation": 90,
                              "color": "red" if m else "blue"}
                    if amp > 0.5 * ymax:
                        ax.text(yval, amp, str(wave), verticalalignment='top',
                                **kwargs)
                    else:
                        ax.text(yval, ymax*0.05, str(wave), **kwargs)
            ax.set_ylim(0, ymax * 1.05)
            ax.set_xlim(0, arm.szy - 1)
            ax.text(20, ymax * 0.8, f'order {order}')
            ax.set_yticks([])  # we don't care about y scale
        for i in range(m_ix % nrows + 1, 4):
            axes[i].axis('off')
        fig.subplots_adjust(hspace=0)
        pdf.savefig(bbox_inches='tight')


def make_wavelength_table(ext):
    """
    Produce an image of the same shape as the data, where each pixel has a
    value corresponding to the wavelength of that pixel. I keep getting this
    wrong because of the astropy and gWCS x-first format.

    Parameters
    ----------
    ext: single-slice AstroData object

    Returns
    -------
    ndarray with ext.shape:
        wavelength of each pixel
    """
    grid = np.meshgrid(*list(np.arange(length) for length in reversed(ext.shape)),
                       sparse=True, indexing='xy')
    return ext.wcs(*grid)


def get_wavelength_limits(ext):
    """
    Determine the shortest and longest wavelengths for pixels in a spectral
    image, by evaulating its WCS at all corners of the image. The wavelength
    axis is assumed to be the first value in the world coordinates, by
    convention.

    Parameters
    ----------
    ext: single-slice AstroData object

    Returns
    -------
    2-tuple: min and max wavelength values
    """
    values = ext.wcs(*reversed(list(zip(*at.get_corners(ext.shape)))))
    if isinstance(values, tuple):
        values = values[0]
    return min(values), max(values)


def model_scattered_light(data, mask=None, variance=None):
    """
    Model scattered light by fitting a surface to the 2D image. The fit is
    performed as the exponential of a 2D polynomial to ensure it is always
    non-negative.

    Parameters
    ----------
    data: `numpy.ndarray`
        data which requires a surface fit
    mask: `numpy.ndarray` (bool)/None
        mask indicating which pixels to use in the fit

    Returns
    -------
    model: `numpy.ndarray`
        a model fit to the data
    """
    from datetime import datetime
    ny, nx = data.shape
    y, x = np.mgrid[:ny, :nx]
    if variance is None:
        w = np.ones_like(data)
    else:
        w = np.sqrt(at.divide0(1., variance))
    start = datetime.now()
    tck = interpolate.bisplrep(x[~mask].ravel()[::32], y[~mask].ravel()[::32], data[~mask].ravel()[::32], w=w[~mask].ravel()[::32])
    print(datetime.now() - start)
    spl = interpolate.bisplev(np.arange(nx), np.arange(ny), tck)
    print(datetime.now() - start)
    return spl
