"""
Given (x,wave,matrices, slit_profile), extract the flux from each order.
"""

from __future__ import division, print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
import matplotlib.cm as cm
import warnings
from scipy.interpolate import CubicSpline
from datetime import datetime

from gempy.library import astrotools, matching
from gempy.utils import logutils
from geminidr.gemini.lookups import DQ_definitions as DQ
from .extractum import Extractum


log = logutils.get_logger(__name__)


class Extractor(object):
    """
    A class to extract data for each arm of the spectrograph.

    The extraction is defined by 3 key parameters: an ``x_map``, which is
    equivalent to 2dFDR's tramlines and contains a physical x-coordinate for
    every y (dispersion direction) coordinate and order, and a ``w_map``,
    which is the wavelength corresponding to every y (dispersion direction)
    coordinate and order.

    The slit parameters are defined by the slitview_instance. If a different
    slit profile is to be assumed, then another (e.g. simplified or fake)
    slitview instance should be made, rather than modifying this class.

    Attributes
    ----------
    arm: :any:`polyspect.Polyspect`
        This defines, e.g., whether the camera is ``"red"`` or ``"blue"``

    slitview: :any:`slitview.SlitView`
        This defines, e.g., whether the mode is ``"std"`` or ``"high"``

    gain: float, optional
        gain in electrons per ADU. From fits header.

    rnoise: float, optional
        Expected readout noise in electrons. From fits header.

    badpixmask: :obj:`numpy.ndarray`, optional
        A data quality plane, which evaluates to False (i.e. 0) for good
        pixels.

    vararray: :obj:`numpy.ndarray`, optional
        A variance array.

    transpose: bool , optional
        Do we transpose the data before extraction? Default is ``False``.

    cr_flag: integer, optional
        When we flag additional cosmic rays in the badpixmask, what value
        should we use? Default is ``8``.
    """
    def __init__(self, polyspect_instance, slitview_instance,
                 gain=1.0, rnoise=3.0, cr_flag=DQ.cosmic_ray,
                 badpixmask=None, transpose=False,
                 vararray=None):
        self.arm = polyspect_instance
        self.slitview = slitview_instance
        self.transpose = transpose
        self.gain = gain
        self.rnoise = rnoise
        self.vararray = vararray
        self.badpixmask = badpixmask
        self.cr_flag = cr_flag

        # FIXME: This warning could probably be neater.
        if not isinstance(self.arm.x_map, np.ndarray):
            raise UserWarning('Input polyspect_instance requires'
                              'spectral_format_with matrix to be run.')

        # To aid in 2D extraction, let's explicitly compute the y offsets
        # corresponding to these x offsets...
        # The "matrices" map pixels back to slit co-ordinates.
        ny = self.arm.x_map.shape[1]
        nm = self.arm.x_map.shape[0]
        self.slit_tilt = np.zeros((nm, ny))
        for i in range(nm):
            for j in range(ny):
                invmat = np.linalg.inv(self.arm.matrices[i, j])
                # What happens to the +x direction?
                x_dir_map = np.dot(invmat, [1, 0])
                self.slit_tilt[i, j] = x_dir_map[1] / x_dir_map[0]

    def bin_models(self):
        """
        Bin the models to match data binning.

        Function used to artificially bin the models so that they apply to
        whatever binning mode the data are. This requires knowledge of the 
        x and y binning from the arm class, which is assumed this class
        inherits. 
        
        The binning is done as a running average, in which the
        values for each binned pixel are assumed to be equivalent to the average
        value of all physical pixels that are part of the binned pixel.

        Returns
        -------
        x_map: :obj:`numpy.ndarray`
            Binned version of the x map model
        wmap: :obj:`numpy.ndarray`
            Binned version of the wavelength map model
        blaze: :obj:`numpy.ndarray`
            Binned version of the blaze map model
        matrices: :obj:`numpy.ndarray`
            Binned version of the matrices array
        """
        if self.arm.xbin == 1 and self.arm.ybin == 1:
            return self.arm.x_map, self.arm.w_map, self.arm.blaze, \
                   self.arm.matrices
        # Start by getting the order number. This should never change.
        n_orders = self.arm.x_map.shape[0]

        x_map = self.arm.x_map.copy()
        w_map = self.arm.w_map.copy()
        blaze = self.arm.blaze.copy()
        matrices = self.arm.matrices.copy()
        # The best way to do this is to firstly do all the ybinning, and then do
        # the x binning
        if self.arm.ybin > 1:
            # Now bin the x_map, firstly in the spectral direction
            # We do this by reshaping the array by adding another dimension of
            # length ybin and then averaging over this axis
            x_map = np.mean(x_map.reshape(n_orders,
                                          int(self.arm.szy / self.arm.ybin),
                                          self.arm.ybin), axis=2)

            # Now do the same for the wavelength scale and blaze where necessary
            w_map = np.mean(w_map.reshape(n_orders,
                                          int(self.arm.szy / self.arm.ybin),
                                          self.arm.ybin), axis=2)

            blaze = np.mean(blaze.reshape(n_orders,
                                          int(self.arm.szy / self.arm.ybin),
                                          self.arm.ybin), axis=2)
            # The matrices are a bit harder to work with, but still the same
            # principle applies.
            matrices = np.mean(matrices.reshape(n_orders,
                                                int(
                                                    self.arm.szy /
                                                    self.arm.ybin),
                                                self.arm.ybin, 2, 2), axis=2)

        if self.arm.xbin > 1:
            # Now, naturally, the actualy x values must change according to the
            # xbin
            x_map = (x_map + 0.5) / self.arm.xbin - 0.5
            # The w_map and blaze remain unchanged by this. 

        # Now we must modify the values of the [0,0] and [1,1] elements of
        # each matrix according to the binning to reflect the now size of
        # binned pixels.
        rescale_mat = np.array([[self.arm.xbin, 0], [0, self.arm.ybin]])
        matrices = np.dot(matrices, rescale_mat)

        # Set up convenience local variables
        ny = x_map.shape[1]
        nm = x_map.shape[0]

        self.slit_tilt = np.zeros((nm, ny))
        for i in range(nm):
            for j in range(ny):
                invmat = np.linalg.inv(matrices[i, j])
                # What happens to the +x direction?
                x_dir_map = np.dot(invmat, [1, 0])
                self.slit_tilt[i, j] = x_dir_map[1] / x_dir_map[0]

        return x_map, w_map, blaze, matrices

    def make_pixel_model(self, input_image=None):
        """
        Based on the xmod and the slit viewer image, create a complete model image, 
        where flux versus wavelength pixel is constant. As this is designed for 
        comparing to flats, normalisation is to the median of the non-zero pixels in the
        profile.
        
        Parameters
        ----------
        input_iage: :obj:`numpy.ndarray`, optional
            Image data, transposed so that dispersion is in the "y" direction.
            If this is given, then the pixel model is scaled according to the input flux
            for every order and wavelength. Note that this isn't designed to reproduce
            dual-object or object+sky data.
        
        Returns
        -------
        model: :obj:`numpy.ndarray`
            An image the same size as the detector.
        """
        # Adjust in case of y binning (never for flats, which is what this function
        # is primarily designed for.
        try:
            x_map, w_map, blaze, matrices = self.bin_models()
        except Exception:
            raise RuntimeError('Extraction failed, unable to bin models.')

        # Key array constants
        ny = x_map.shape[1]
        nm = x_map.shape[0]
        nx = int(self.arm.szx / self.arm.xbin)
                             
        profiles = [self.slitview.slit_profile(arm=self.arm.arm)]
        
        n_slitpix = profiles[0].shape[0]
        profile_y_microns = (np.arange(n_slitpix) -
                             n_slitpix / 2 + 0.5) * self.slitview.microns_pix
        
        if self.transpose:
            pixel_model = np.zeros((ny, nx))
        else:
            pixel_model = np.zeros((nx, ny))
        
        # Loop through all orders then through all y pixels.        
        print("    Creating order ", end="")
        for i in range(nm):
            print(f"{self.arm.m_min+i}...", end="")
            sys.stdout.flush()

            # Create an empty model array. Base the size on the largest
            # slit magnification for this order.
            nx_cutout = int(np.ceil(self.slitview.slit_length / np.min(
                matrices[i, :, 0, 0])))

            for j in range(ny):
                x_ix, phi, profiles = resample_slit_profiles_to_detector(
                    profiles, profile_y_microns, x_map[i, j] + nx // 2,
                    detpix_microns=matrices[i, j, 0, 0])

                # This isn't perfect as it gets the top pixel wrong if the
                # trace goes over the top but it's OK for this purpose
                if self.transpose:
                    pixel_model[j, np.minimum(x_ix, nx-1)] = phi[0]
                else:
                    pixel_model[np.minimum(x_ix, nx-1), j] = phi[0]
                    
        return pixel_model

    def new_extract(self, data=None, correct_for_sky=True, use_sky=True,
                    optimal=False, find_crs=True, snoise=0.1, sigma=6,
                    used_objects=[0, 1], debug_pixel=None,
                    apply_centroids=False, correction=None, ftol=0.001,
                    timing=False):
        """
        Do a complete extraction of all objects from the echellogram.

        This function also identifies cosmic rays if requested. Since both
        the extraction and CR require a noise model, the two steps have
        been combined here to avoid duplication of code.

        Parameters
        ----------
        data: :obj:`numpy.ndarray`
            Image data, transposed so that dispersion is in the "y" direction.
            Note that this is the transpose of a conventional echellogram.
        correct_for_sky: bool, optional
            Do we correct the object slit profiles for sky? Should be yes for
            objects and no for flats/arcs.
        use_sky: book, optional
            In extraction, do we use sky (and therefore self-subtract)?
        optimal: bool
            perform optimal (rather than uniform) extraction?
        find_crs : bool
            identify cosmic rays in the data? This is achieved by comparing
            vertical cuts with the slit profile and does not account for
            slit tilt
        snoise : float
            linear fraction of signal to add to noise estimate for CR flagging
        sigma : float
            number of standard deviations for identifying discrepant pixels
        used_objects : list
            from which IFUs should sources be extracted?
        debug_pixel : 2-tuple
            (order, pixel) of extraction to plot for debugging purposes
        apply_centroids : bool
            apply shifts based on the transverse location of the centre of
            light in the slitviewer image? CRH suggests that this just adds
            noise and decreases the resolution for sources of typical S/N
        correction: None/:obj:`numpy.ndarray`
            blaze correction factor to apply to each pixel in each order
        ftol: float
            fractional tolerance for convergence

        Returns
        -------
        extracted_flux: :obj:`numpy.ndarray`
            Extracted fluxes as a function of pixel along the spectral direction
        extracted_var: :obj:`numpy.ndarray`
            Extracted variance as a function of pixel along the spectral
            direction
        """
        try:
            x_map, w_map, blaze, matrices = self.bin_models()
        except Exception:
            raise RuntimeError('Extraction failed, unable to bin models.')

        # Set up convenience local variables
        ny = x_map.shape[1]
        nm = x_map.shape[0]
        nx = int(self.arm.szx / self.arm.xbin)

        if self.badpixmask is None:
            self.badpixmask = np.zeros_like(data, dtype=DQ.datatype)

        # Our profiles... we re-extract these in order to include the centroids
        # Centroids is the offset in pixels along the short axis of the pseudoslit
        profile, centroids = self.slitview.slit_profile(arm=self.arm.arm,
                                                        return_centroid=True)

        # Allow us to compute flux-weighted transverse position
        slitview_profiles = [profile, profile*centroids]

        def transverse_positions(slitview_profiles, profile_center=None,
                                 detpix_microns=None):
            """Returns offset along short axis of pseudoslit in microns"""
            ix, p, _ = resample_slit_profiles_to_detector(
                slitview_profiles, profile_y_microns=profile_y_microns,
                profile_center=profile_center, detpix_microns=detpix_microns)
            return ix, astrotools.divide0(p[1], p[0]) * self.slitview.microns_pix

        profiles = self.slitview.object_slit_profiles(
            arm=self.arm.arm, correct_for_sky=correct_for_sky,
            used_objects=used_objects, append_sky=use_sky or find_crs
        )

        # Number of "objects" and "slit pixels"
        no = profiles.shape[0]
        if find_crs and not use_sky:
            no -= 1
        n_slitpix = profiles.shape[1]
        profile_y_microns = (np.arange(n_slitpix) -
                             n_slitpix / 2 + 0.5) * self.slitview.microns_pix

        m_init = models.Polynomial1D(degree=1)
        fit_it = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(), sigma_clip)
        good = ~np.isinf(self.vararray) & ((self.badpixmask & DQ.not_signal) == 0)
        m_noise, _ = fit_it(m_init, data[good].ravel(), self.vararray[good].ravel())
        if m_noise.c0 <= 0:
            log.warning("Problem with read noise estimate!")
            log.warning(m_noise)
            m_noise.c0 = 4  # just something for now
            m_noise.c1 = 1
        noise_model = lambda x: m_noise.c0 + m_noise.c1 * abs(x)

        pixel_inv_var = 1. / self.vararray

        # Identify CRs
        saturation_warning = False
        if find_crs:
            print("    Finding CRs in order ", end="")
            for i in range(nm):
                print(f"{self.arm.m_min+i}...", end="")
                sys.stdout.flush()

                for j in range(ny):
                    debug_this_pixel = debug_pixel in [(self.arm.m_min+i, j)]

                    x_ix, phi, profiles = resample_slit_profiles_to_detector(
                        profiles, profile_y_microns, x_map[i, j] + nx // 2,
                        detpix_microns=matrices[i, j, 0, 0],
                        debug=debug_this_pixel)

                    # Deal with edge effects...
                    ww = np.logical_or(x_ix >= nx, x_ix < 0)
                    x_ix[ww] = 0
                    phi /= phi.sum(axis=1)[:, np.newaxis]

                    _slice = (j, x_ix) if self.transpose else (x_ix, j)
                    col_data = data[_slice]
                    col_inv_var = pixel_inv_var[_slice]
                    badpix = self.badpixmask[_slice].astype(bool)
                    badpix[ww] = True
                    xtr = Extractum(phi, col_data, col_inv_var, badpix,
                                    noise_model=noise_model,
                                    pixel=(self.arm.m_min+i, j))
                    if debug_this_pixel:
                        print("ROWS", x_ix)
                    xtr.find_cosmic_rays(snoise=snoise, sigma=sigma,
                                         debug=debug_this_pixel)
                    # Do NOT flag bad pixels as CRs
                    self.badpixmask[_slice] |= (
                            (xtr.cr & ~(self.badpixmask[_slice] &
                                        DQ.bad_pixel).astype(bool)) * DQ.cosmic_ray)
                    if debug_this_pixel:
                        print("ADDING CRs", _slice)
                        print(self.badpixmask[_slice])
                    if (not saturation_warning and
                            np.any(self.badpixmask[_slice] &
                                   (DQ.cosmic_ray | DQ.saturated |
                                    DQ.bad_pixel) == DQ.saturated)):
                        print("\n")
                        log.warning("There are saturated pixels that have not "
                                    "been flagged as cosmic rays in order "
                                    f"{self.arm.m_min+i} pixel {j} "
                                    f"({x_ix[0]}-{x_ix[-1]})")
                        saturation_warning = True
            print("\n")
            log.stdinfo(f"{(self.badpixmask & DQ.cosmic_ray).astype(bool).sum()} CRs found")
            if self.arm.mode == 'high':
                log.stdinfo("(due to scattered light, the topmost pixel in HR"
                            " is often incorrectly flagged)")

        # Now do the extraction. First determine *where* to extract
        extracted_flux = np.zeros((nm, ny, no), dtype=np.float32)
        extracted_var = np.zeros((nm, ny, no), dtype=np.float32)
        start = datetime.now()
        profiles = profiles[:no]
        print("\n\n    Extracting order ", end="")
        c0, c1 = m_noise.parameters
        for i in range(nm):
            print(f"{self.arm.m_min+i}...", end="")
            sys.stdout.flush()

            # Determine which rows have data from this order
            for j in (0, ny-1):
                slit_center = x_map[i, j] + nx // 2
                x_ix, phi, profiles = resample_slit_profiles_to_detector(
                    profiles, profile_y_microns, slit_center,
                    detpix_microns=matrices[i, j, 0, 0])
                if j == 0:
                    limits = (x_ix.min(), x_ix.max())
                elif x_ix.min() < limits[0]:
                    xmin, xmax = x_ix.min(), limits[1]
                else:
                    xmin, xmax = limits[0], x_ix.max()

            # xmin can be <0, xmax can be >= nx
            nrows = xmax - xmin + 1
            pixel_array = np.zeros((ny, nrows))
            mask_array = np.zeros_like(pixel_array, dtype=bool)
            all_phi = []
            if debug_pixel != (None, None):
                print(f"ROW LIMITS: {xmin} {xmax}")

            # Code is written in this way to minimize the number of calls to
            # np.interp -- calling for each pixel is very slow
            for j in range(ny):
                debug_this_pixel = debug_pixel in [(self.arm.m_min+i, j)]
                slit_center = x_map[i, j] + nx // 2
                x_ix, phi, profiles = resample_slit_profiles_to_detector(
                    profiles, profile_y_microns, slit_center,
                    detpix_microns=matrices[i, j, 0, 0])

                # Deal with edge effects...
                on_array = np.logical_and(x_ix >=0, x_ix < nx)
                phi /= phi.sum(axis=1)[:, np.newaxis]

                # Calculate extraction location for each spatial pixel by
                # including contributions from non-collinear slit and slit tilt
                ytilt = (x_ix - slit_center) * self.slit_tilt[i, j]
                y_ix = j + ytilt
                if apply_centroids:
                    ix, xvpos = transverse_positions(slitview_profiles, slit_center,
                                                     detpix_microns=matrices[i, j, 0, 0])
                    assert np.array_equal(x_ix, ix)
                    y_ix += xvpos / matrices[i, j, 1, 1] + ytilt
                pixel_array[j, x_ix.min()-xmin:x_ix.max()-xmin+1] = y_ix
                if debug_this_pixel:
                    y_locations = y_ix
                    print("\nY LOCATIONS",
                          [(xx, yy) for xx, yy in zip(x_ix, y_locations)])
                mask_array[j, x_ix.min()-xmin:x_ix.max()-xmin+1] |= ~on_array
                all_phi.append((x_ix.min()-xmin, phi))

            if debug_pixel[0] == self.arm.m_min+i:
                yy = debug_pixel[1]
                x1, phi = all_phi[yy]
                log.debug(f"Data and badpix around ({x1+xmin}:{x1+xmin+phi.shape[1]}, {yy})")
                for xx in range(x1+xmin, x1+xmin+phi.shape[1]):
                    log.debug(xx, data[xx, yy-1:yy+2], self.badpixmask[xx, yy-1:yy+2])
                if correction is not None:
                    log.debug("Correction factors (multiplicative)")
                    log.debug(correction[i, yy-1:yy+2])

            # Do the interpolation for all wavelengths in this order
            # Save memory by overwriting the array of pixel locations with values
            mask_array |= np.logical_or(pixel_array < 0, pixel_array >= ny)
            for ix in range(max(xmin, 0), min(xmax+1, nx)):
                # Flag any virtual pixel that is partly off the edge of the array
                mask_array[:, ix-xmin] |= np.interp(
                    pixel_array[:, ix-xmin], np.arange(ny), self.badpixmask[ix]) > 0
                if correction is None:
                    pixel_array[:, ix-xmin] = np.interp(
                        pixel_array[:, ix-xmin], np.arange(ny), data[ix])
                else:
                    pixel_array[:, ix-xmin] = np.interp(
                        pixel_array[:, ix-xmin],
                        np.arange(ny), data[ix] * correction[i])

            if debug_pixel[0] == self.arm.m_min+i:
                for ix, y in enumerate(y_locations):
                    log.debug(ix+x1+xmin, y,  pixel_array[yy, ix+x1], mask_array[yy, ix+x1])

            for j, (x_ix_min, phi) in enumerate(all_phi):  # range(ny)
                debug_this_pixel = debug_pixel in [(self.arm.m_min+i, j)]
                if correction is not None:
                    if correction[i, j] == 0:
                        extracted_flux[i, j] = 0
                        extracted_var[i, j] = 0
                        if debug_this_pixel:
                            log.debug("FLATFIELD CORRECTION IS ZERO")
                        continue
                    c0, c1 = correction[i, j] ** 2 * m_noise.parameters
                    # Linear term only needs to be multiplied by a single
                    # factor of the blaze correction because the data have
                    # already been multiplied by a factor 20 lines above.
                    noise_model = lambda x: c0 + c1 * abs(x) / correction[i, j]

                _slice = (j, slice(x_ix_min, x_ix_min+phi.shape[1]))
                xtr = Extractum(phi, pixel_array[_slice],
                                mask=mask_array[_slice], noise_model=noise_model,
                                pixel=(self.arm.m_min+i, j))
                model_amps = xtr.fit(debug=debug_this_pixel, c0=c0,
                                     c1=c1, ftol=ftol)

                phi_scaled = phi * model_amps[:, np.newaxis]
                sum_models = phi_scaled.sum(axis=0)
                col_var = noise_model(sum_models)  # definitely +ve everywhere
                frac = astrotools.divide0(phi_scaled, sum_models)
                frac[:, xtr.mask] = 0

                # Optimally-extracted flux is well-defined. Uniform extraction
                # is not so clear. Variance is a bit tricky; we use the
                # formula for VAR(f)/f in Table 1 of Horne (1986)
                if optimal:
                    extracted_flux[i, j] = model_amps
                    extracted_var[i, j] = abs(
                        extracted_flux[i, j] * astrotools.divide0(
                            phi[:, ~xtr.mask].sum(axis=1),
                            (abs(xtr.data - sum_models + phi_scaled) * phi / col_var)[:, ~xtr.mask].sum(axis=1)))
                else:
                    # Correction for flagged pixels
                    object_scaling = astrotools.divide0(phi.sum(axis=1),
                                                        phi[:, ~xtr.mask].sum(axis=1))
                    extracted_flux[i, j] = np.dot(frac, xtr.data) * object_scaling
                    extracted_var[i, j] = np.dot(frac, col_var) * object_scaling ** 2

                if debug_this_pixel:
                    log.debug("EXTRACTED FLUXES", extracted_flux[i, j])
                    log.debug("EXTRACTED VAR", extracted_var[i, j])
            if timing:
                print(datetime.now() - start)
        print("\n")

        return extracted_flux, extracted_var

    def quick_extract(self, data=None):
        """
        Extract flux by simply summing all the pixels at each column
        in each order. This is used in the measureBlaze() primitive
        to determine the relative sensitivity at each wavelength.
        """
        try:
            x_map, w_map, blaze, matrices = self.bin_models()
        except Exception:
            raise RuntimeError('Extraction failed, unable to bin models.')

        # Set up convenience local variables
        nm, ny = x_map.shape
        nx = int(self.arm.szx / self.arm.xbin)

        extracted_flux = np.zeros((nm, ny), dtype=np.float32)
        extracted_mask = np.zeros_like(extracted_flux, dtype=DQ.datatype)
        extracted_var = np.zeros_like(extracted_flux)

        profiles = [self.slitview.slit_profile(arm=self.arm.arm)]
        n_slitpix = profiles[0].size
        profile_y_microns = (np.arange(n_slitpix) -
                             n_slitpix / 2 + 0.5) * self.slitview.microns_pix

        # Loop through all orders then through all y pixels.
        print("    Extracting order ", end="")
        for i in range(nm):
            print(f"{self.arm.m_min+i}...", end="")
            sys.stdout.flush()

            for j in range(ny):
                x_ix, phi, profiles = resample_slit_profiles_to_detector(
                    profiles, profile_y_microns, x_map[i, j] + nx // 2,
                    detpix_microns=matrices[i, j, 0, 0])
                _slice = (j, x_ix) if self.transpose else (x_ix, j)
                if x_ix.min() < 0 or x_ix.max() >= nx:
                    extracted_mask[i, j] = DQ.no_data
                else:
                    extracted_flux[i, j] = data[_slice].sum()
                    extracted_var[i, j] = self.vararray[_slice].sum()
                    extracted_mask[i, j] = np.logical_or.reduce(self.badpixmask[_slice])

        print("\n")
        if np.any(extracted_mask & DQ.saturated):
            log.warning("Some pixels are saturated")
        elif np.any(extracted_mask & DQ.non_linear):
            log.warning("Some pixels are in the non-linear regime")

        return extracted_flux, extracted_var, extracted_mask

    def find_lines(self, flux, arclines, hw=12,
                   arcfile=None, # Now dead-letter - always overridden
                   inspect=False, plots=False):
        """
        THIS FUNCTION IS NO LONGER USED!

        Find lines near the locations of input arc lines.
        
        This is done with Gaussian fits near the location of where lines are
        expected to be. An initial decent model must be present, likely
        the result of a manual adjustment.

        Parameters
        ----------
        flux: :obj:`numpy.ndarray`
            Flux data extracted with the 1D extractor. Just the flux, not the
            variance.

        arclines: float array
            Array containing the wavelength of the arc lines.


        arcfile: float array
            Arc file data.

        inspect: bool, optional
            If true, show display of lines and where they are predicted to fall

        plots: bool, optional
            If true, plot every gaussian fit along the way for visual inspection 

        Returns
        -------

        lines_out: float array
            Whatever used to be placed in a file.
        """
        # arcfile = flux
        # Only use the middle object.
        # In High res mode this will be the object, in std mode it's the sky
        flux = flux[:, :, 0]
        arcfile = flux
        ny = flux.shape[1]
        nm = flux.shape[0]
        nx = self.arm.szx
        lines_out = []
        # Let's try the median absolute deviation as a measure of background
        # noise if the search region is not large enough for robust median
        # determination.
        if hw < 8:
            noise_level = np.median(np.abs(flux - np.median(flux)))

        if (inspect == True) and (arcfile is None):
            raise UserWarning('Must provide an arc image for the inpection')
        if inspect or plots:
            image = np.arcsinh((arcfile - np.median(arcfile)) / 1e2)
        if inspect:
            plt.imshow(image, interpolation='nearest', aspect='auto',
                       cmap=cm.gray)

        for m_ix in range(nm):
            # Select only arc lines that should be in this order.
            filtered_arclines = arclines[
                (arclines >= self.arm.w_map[m_ix, :].min())
                & (arclines <= self.arm.w_map[m_ix, :].max())]
            # This line interpolates between filtered lines and w_map on a
            # linear array, to find the expected pixel locations of the lines.
            w_ix = np.interp(filtered_arclines, self.arm.w_map[m_ix, :],
                             np.arange(ny))
            # Ensure that lines close to the edges of the chip are not
            # considered
            ww = np.where((w_ix >= hw) & (w_ix < ny - hw))[0]
            w_ix = w_ix[ww]
            arclines_to_fit = filtered_arclines[ww]
            log.stdinfo('order ', m_ix)
            for i, ix in enumerate(w_ix):
                # This ensures that lines too close together are not used in the
                # fit, whilst avoiding looking at indeces that don't exist.
                if len(w_ix) > 1:
                    if (np.abs(ix - w_ix[i - 1]) < 1.5 * hw):
                        continue
                    elif i != (len(w_ix) - 1) and (
                            np.abs(ix - w_ix[i + 1]) < 1.5 * hw):
                        continue
                x = np.arange(ix - hw, ix + hw, dtype=int)
                y = flux[m_ix, x]
                # Try median absolute deviation for noise characteristics if
                # Enough pixels are available per cut.
                if hw >= 7:
                    noise_level = np.median(np.abs(y - np.median(y)))
                # Any line with peak S/N under a value is not considered.
                # e.g. 20 is considered.
                if (np.max(y) < 20 * noise_level):
                    log.warning("Rejecting due to low SNR!")
                    continue

                g_init = models.Gaussian1D(amplitude=np.max(y), mean=x[
                    np.argmax(y)], stddev=1.5)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit_g = fitting.LevMarLSQFitter()
                    g = fit_g(g_init, x, y)
                # Wave, ypos, xpos, m, amplitude, fwhm
                xpos = nx // 2 + \
                       np.interp(g.mean.value, np.arange(ny),
                                 self.arm.x_map[m_ix])
                ypos = g.mean.value

                line_to_append = [arclines_to_fit[i], ypos, xpos, m_ix +
                                  self.arm.m_min, g.amplitude.value,
                                  g.stddev.value * 2.3548]

                # If any of the values to append are nans, don't do it and
                # just go on to the next line.
                if np.isnan(line_to_append).any():
                    continue

                # This option is here to allow the user to inspect individual
                # gaussian fits. Useful to test whether the method is working.
                if plots:
                    f, sub = plt.subplots(1, 2)
                    sub[0].plot(x, y)
                    sub[0].plot(x, g(x))
                    sub[0].axvline(ix)
                    snapshot = image[int(ix - hw * 4):int(ix + hw * 4),
                               int(xpos - 40):
                               int(xpos + 40)]
                    sub[1].imshow(np.arcsinh((snapshot - np.median(snapshot)) /
                                             1e2))
                    plt.show()

                # This part allows the user to inspect the global fit and
                # position finding. 
                if inspect:
                    plt.plot(xpos, ix, 'bx')
                    plt.plot(xpos, ypos, 'rx')
                    plt.text(xpos + 10, ypos,
                             str(arclines_to_fit[i]), color='green',
                             fontsize=10)

                lines_out.append(line_to_append)

        if inspect:
            plt.axis([0, nx, ny, 0])
            plt.show()

        return np.array(lines_out)

    def match_lines(self, all_peaks, arclines, hw=12, xcor=False, log=None):
        """
        Match peaks in data to arc lines based on proximity

        Parameters
        ----------
        all_peaks: list of :obj:`numpy.ndarray`
            Locations of peaks in each order

        arclines: float array
            Array containing the wavelength of the arc lines.

        hw: int, optional
            Number of pixels from each order end to be ignored due to proximity
            with the edge of the chip. Default was 10.

        xcor: bool
            Perform initial cross-correlation to find gross shift?

        log: logger/None

        Returns
        -------
        lines_out: 2D array
            arc line wavelength, fitted position, position along orthogonal
            direction, order, amplitude, FWHM
        """
        nx, ny = self.arm.szx, self.arm.szy
        pixels = np.arange(ny)
        lines_out = []

        for m_ix, peaks in enumerate(all_peaks):
            filtered_arclines = arclines[
                (arclines >= self.arm.w_map[m_ix].min())
                & (arclines <= self.arm.w_map[m_ix].max())]
            # This line interpolates between filtered lines and w_map on a
            # linear array, to find the expected pixel locations of the lines.
            w_ix = np.interp(filtered_arclines, self.arm.w_map[m_ix, :], pixels)
            # Ensure that lines close to the edges of the chip are not
            # considered
            ww = np.where((w_ix >= hw) & (w_ix < ny - hw))[0]
            if ww.size * len(peaks) == 0:
                continue

            w_ix = w_ix[ww]
            arclines_to_fit = filtered_arclines[ww]
            if log:
                log.stdinfo(f'order {m_ix+self.arm.m_min:2d} with '
                            f'{len(peaks):2d} peaks and {ww.size:2d} arc lines')

            # Perform cross-correlation if requested
            if xcor:
                expected = np.zeros((ny,)) + np.sum([np.exp(-(cntr - pixels) ** 2 / 18)
                                   for cntr in w_ix], axis=0)  # stddev=3
                synth = np.zeros((ny,)) + np.sum([np.exp(-(g.mean.value - pixels) ** 2 / 18)
                                for g in peaks], axis=0)  # stddev=3
                # -ve means line peaks are "ahead" of expected
                shift = np.correlate(expected, synth, mode='full')[ny-hw-1:ny+hw].argmax() - hw
            else:
                shift = 0

            xpos = lambda g: nx // 2 + np.interp(g.mean.value, pixels,
                                                 self.arm.x_map[m_ix])
            matched = matching.match_sources([g.mean.value for g in peaks],
                                             w_ix - shift, radius=hw)
            new_lines = [(arclines_to_fit[m], g.mean.value, xpos(g), m_ix + self.arm.m_min,
                          g.amplitude.value, g.stddev.value * 2.3548)
                         for i, (m, g) in enumerate(zip(matched, peaks)) if m > -1]
            lines_out.extend(new_lines)

        return np.array(lines_out)


def resample_slit_profiles_to_detector(profiles, profile_y_microns=None,
                                       profile_center=None, detpix_microns=None,
                                       debug=False):
    """
    This code takes one or more 1D profiles from the slitviewer camera and
    resamples them onto the (coarser) sampling of the echellograms. Because we
    are resampling a discretely-sampled profile onto another discrete pixel
    grid, "ringing" effects abound and so, in an attempt to reduce these, the
    profiles are expressed as cubic splines, with this function converting
    arrays into CubicSpline objects and returning those for later use (to
    prevent repeated computations; for these reason, the splines are expressed
    as functions of the offset from the centre of the slit in microns, so the
    pixel scale of the echellogram must be passed to this function).

    Parameters
    ----------
    profiles: iterable (length M) of arrays (N pixels) or spline objects
        profiles of each of the M objects, as measured on the slitviewer camera
    profile_y_microns: ndarray
        locations of each slitviewer pixel in microns, relative to slit center
    profile_center: float
        detector pixel location of center of slit (from XMOD)
    detpix_microns: float
        size of each detector pixel in microns (from SPATMOD)

    Returns
    -------
    x_ix: array of ints, of shape (P,)
        pixels in the spatial direction of the echellogram where the resampled
        profiles are to be placed
    phi: MxP ndarray
        profiles in resampled detector pixel space
    profiles: list of CubicSpline objects
        the profiles of the objects expressed as cubic splines (if they were not
        already)
    """
    # Must turn an array into a list so items can be replaced with CubicSplines
    # if they're not already
    if not isinstance(profiles, list):
        profiles = list(profiles)

    half_slitprof_pixel = 0.5 * np.diff(profile_y_microns).mean()

    # To avoid repeated computations, we recast each profile as a CubicSpline
    # with x being the separation from the slit center in *microns*. The
    # splines only extend over the region of data, to ensure they're precisely
    # zero away from this.
    for i, profile in enumerate(profiles):
        if not isinstance(profile, CubicSpline):
            # There's probably a better way to do this
            good = np.ones_like(profile, dtype=bool)
            for j, val in enumerate(profile):
                if val == 0:
                    good[j] = False
                else:
                    break
            for j, val in enumerate(profile[::-1], start=1):
                if val == 0:
                    good[-j] = False
                else:
                    break
            xspline = np.r_[profile_y_microns[good].min() - half_slitprof_pixel,
                            profile_y_microns[good],
                            profile_y_microns[good].max() + half_slitprof_pixel]
            # Apparently the antiderivative is also a CubicSpline object
            profiles[i] = CubicSpline(xspline, np.r_[0, profile[good], 0]).antiderivative()

    slitview_pixel_edges = profile_center + np.linspace(
        profile_y_microns[0] - half_slitprof_pixel,
        profile_y_microns[-1] + half_slitprof_pixel,
        profile_y_microns.size + 1) / detpix_microns
    x_ix = np.round([slitview_pixel_edges.min(),
                     slitview_pixel_edges.max() + 1]).astype(int)
    pixel_edges_microns = (np.arange(x_ix[0], x_ix[1]+0.1) - 0.5 -
                           profile_center) * detpix_microns

    phi = np.zeros((len(profiles), x_ix[1] - x_ix[0]))
    for i, profile in enumerate(profiles):
        left = np.minimum(np.maximum(pixel_edges_microns[:-1], profile.x[0]), profile.x[-1])
        boundaries = np.r_[left, max(min(pixel_edges_microns[-1], profile.x[-1]), profile.x[0])]
        phi[i] = np.diff(profile(boundaries))

    if debug:
        print("\nRESAMPLING")
        for j, profile in enumerate(profiles):
            print(f"PROFILE {j}")
            print(profile.x)
            print(profile(profile.x))
            print("-"*60)
            #print(pixel_edges_microns)
            #print(profile(pixel_edges_microns))
            #print("="*60)
            for i in range(phi.shape[1]):
                x1 = min(max(pixel_edges_microns[i], profile.x[0]), profile.x[-1])
                x2 = max(min(pixel_edges_microns[i+1], profile.x[-1]), profile.x[0])
                #print(x1, x2, profile.integrate(x1, x2))
            print(phi[j])
        print("="*60)

    phi[phi < 0] = 0
    return np.arange(*x_ix, dtype=int), phi, profiles
