from abc import ABC, abstractmethod

import numpy as np
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
from astropy import units as u

from gempy.library import astrotools as at
from gempy.library.fitting import fit_1D

from .telluric import A0Spectrum, TelluricModels, TelluricSpectrum
from .telluric_models import MultipleTelluricModels, Planck

from datetime import datetime


class Calibrator(ABC):
    """
    An abstract base class for the "Calibrators".

    This is written such that everything is a list of arrays, allowing
    there to be more than one thing being fit. Probably if there's only
    1 thing we want arrays to be returned, not packaged as a list.
    """

    @abstractmethod
    def __len__(self):
        """Number of simultaneous fits, i.e., number of panels in Visualizer"""

    @property
    @abstractmethod
    def x(self):
        """Needs to be defined for 'regions' to have meaning"""

    def set_fitting_params(self, ui_params):
        """
        Set parameters for computing the model, from a single parameter set
        """
        self.reinit_params = {p: getattr(ui_params, p)
                              for p in ui_params.reinit_params}
        fit_params = fit_1D.translate_params(ui_params.values)
        self.fit_params = {k: [v] * len(self) for k, v in fit_params.items()}

    def initialize_user_mask(self, ui_params=None):
        """
        This is the full size of the data, so the Visualizer has to account
        for initially-masked points when updating it.

        This sets the mask initially to empty and then implements any
        user-defined regions from the 'regions' parameter.
        """
        x = self.x
        if isinstance(x, list):
            self.user_mask = [np.zeros_like(x[i], dtype=bool)
                              for i in range(len(self))]
        else:
            self.user_mask = np.zeros_like(x, dtype=bool)

        try:
            regions = ui_params['regions']
        except (TypeError, KeyError):
            return

        # Effectively the code from fit_1D
        if isinstance(regions, str):
            slices = at.parse_user_regions(regions, dtype=np.float64)
        elif isinstance(regions, slice):
            slices = [regions]
        elif all(isinstance(item, slice) for item in regions):
            slices = regions

        # Handle a list of arrays or an array
        if isinstance(x, list):
            for xx, xmask in zip(x, self.user_mask):
                xmask |= at.create_mask_from_regions(xx, slices)
        else:
            self.user_mask |= at.create_mask_from_regions(x, slices)

    @abstractmethod
    def reconstruct_points(self, *args, **kwargs):
        """Create the points"""

    @abstractmethod
    def perform_fit(self, index, sigma_clipping=None):
        """Perform the fit on the specified index"""

    def perform_all_fits(self):
        """Likely default behaviour"""
        return [self.perform_fit(i) for i in range(len(self))]


class TelluricCalibrator(Calibrator):
    """
    The class that does all the heavy lifting
    """
    def __init__(self, telluric_spectra, ui_params=None):
        # Lots of checking that all the TelluricSpectra have similar PCA models
        assert all(isinstance(t, TelluricSpectrum) for t in telluric_spectra)
        self.spectra = telluric_spectra
        npca = [t.pca.npca_params for t in telluric_spectra]
        self.npca = npca[0]
        assert npca == [self.npca] * len(npca)  # all must have the same PCA class
        fixed = [t.pca.fixed for t in telluric_spectra]
        assert fixed == [fixed[0]] * len(fixed)  # all must have the same fixed parameters
        fixed = fixed[0]
        self.lsf_parameter_bounds = {}
        if not fixed:
            if len(telluric_spectra) > 1:
                points = [np.hstack(t.pca.interpolation_points) for t in telluric_spectra]
                assert np.allclose(points, [points[0]] * len(points))
            lsf_params = telluric_spectra[0].lsf.parameters
            self.lsf_parameter_bounds = {param: (np.min(pts), np.max(pts))
                                 for param, pts in zip(lsf_params, telluric_spectra[0].pca.interpolation_points)}
        self.npixels = np.array([tspek.waves.size for tspek in self.spectra])

        # Concatenate the PCA models into a single array now
        #pca_components = np.empty(self.spectra[0].pca.components.shape[:-1] +
        #                          (self.npixels.sum(),))
        #start = 0
        #for tspek, npix in zip(self.spectra, self.npixels):
        #    _slice = slice(start, start+npix)
        #    if fixed:
        #        pca_components[..., _slice] = tspek.pca.components
        #    else:
        #        pca_components[..., _slice] = tspek.pca.interpolation_values
        #    start += npix
        #if fixed:
        #    self.pca = PCA(pca_components)
        #else:
        #    self.pca = PCA(ArrayInterpolator(points=points, values=pca_components))

        # Estimate spectral resolution to which A0 spectrum can be resampled
        self.resolution = 10 * np.max([(tspek.waves[:-1] / np.diff(tspek.waves)).max()
                                       for tspek in self.spectra])

        # We store this as an attribute here since it's not handled fully
        # by TelluricSpectrum (you can make the mask but it's not applied)
        # and it needs to be accessed by the Visualizer
        self.stellar_mask = [tspek.make_stellar_mask(r=self.resolution)
                             for tspek in self.spectra]

        # Allow instantiation without this and can call later
        if ui_params is not None:
            if not fixed:
                for tspek in self.spectra:
                    tspek.pca.set_interpolation_parameters(
                        [getattr(ui_params, k) or self.lsf_parameter_bounds[k][0]
                         for k in lsf_params])
            self.set_fitting_params(ui_params)
            self.reconstruct_points()

        self.initialize_user_mask(ui_params=ui_params)

    def __len__(self):
        return len(self.spectra)

    @property
    def x(self):
        return [tspek.waves for tspek in self.spectra]

    @property
    def y(self):
        return [tspek.data for tspek in self.spectra]

    @property
    def mask(self):
        return [tspek.mask.astype(bool) for tspek in self.spectra]

    def reconstruct_points(self, *args, **kwargs):
        print("REINIT", self.reinit_params)
        magnitude = self.reinit_params["magnitude"]
        w0, f0 = at.Magnitude(magnitude, abmag=self.reinit_params["abmag"]).properties()
        if w0 is None:
            raise RuntimeError(f"Cannot parse magnitude '{magnitude}'")
        wspek, fspek = self.create_stellar_spectrum(w0, f0, self.reinit_params["bbtemp"])

        from datetime import datetime
        start = datetime.now()
        lsf_kwargs = {}
        for tspek in self.spectra:
            tspek.set_intrinsic_spectrum(wspek, fspek, **lsf_kwargs)
            tspek.pca.set_interpolation_parameters([self.reinit_params[k]
                                                    for k in tspek.lsf.parameters])
        print(datetime.now() - start, "INTRINSIC DONE")

    def initialize_user_mask(self, ui_params=None):
        """We put the stellar mask into the user mask"""
        super().initialize_user_mask(ui_params=ui_params)
        for mask, smask in zip(self.user_mask, self.stellar_mask):
            mask |= smask

    def perform_fit(self, index, sigma_clipping=None):
        """
        Perform a telluric fit of (nominally) one spectrum. However, since
        the telluric model is the same across all spectra, this must perform
        a simultaneous fit to all spectra.

        Parameters
        ----------
        index: int
            The index of the spectrum to fit (or the panel in the Visualizer
            that prompted the fit)
        sigma_clipping: bool/None
            Whether to perform sigma-clipping. This is passed by the UI; if
            running non-interactively, then it can be inferred from the value
            of "niter" (sigma_clipping = niter > 0)

        Returns
        -------
        m_final: MultipleTelluricModels
            the best-fitting model
        mask: array (bool)
            points masked in the fit
        """
        # Extract the fitting parameters for this panel
        params = {k: v[index] for k, v in self.fit_params.items()}

        # Start by updating fitting parameters to the value in the
        # panel that called this method
        for p in self.fit_params:
            if p not in ("function", "order"):
                self.fit_params[p] = [params[p]] * len(self)

        return self.perform_all_fits(sigma_clipping=sigma_clipping)

    def perform_all_fits(self, sigma_clipping=None):
        """"""
        print("CALIBRATOR.PERFORM"+"-"*40)
        data = self.concatenate('data')
        mask = self.concatenate('mask').astype(bool)
        original_masks = [tspek.mask.copy() for tspek in self.spectra]
        for tspek, user_mask in zip(self.spectra, self.user_mask):
            tspek.nddata.mask |= user_mask
        print("MASKED PIXEL TOTALS", [tspek.mask.astype(bool).sum()
                                      for tspek in self.spectra])
        m_init = MultipleTelluricModels(
            self.spectra, function=self.fit_params["function"],
            order=self.fit_params["order"])

        # Set the bounds and make sure the initial values are within bounds
        # (because we didn't set the default parameter values). The behaviour
        # is NOT to modify the LSF scaling parameters once the GUI is active
        # or if the user has provided values.
        m_init.bounds.update(self.lsf_parameter_bounds)
        print("REINIT PARAMS", self.reinit_params)
        for k, v in self.lsf_parameter_bounds.items():
            print("INITIALIZING FIT", k, v, self.reinit_params[k])
            if self.reinit_params[k] is None:
                setattr(m_init, k, v[0])
            else:
                setattr(m_init, k, self.reinit_params[k])
                m_init.fixed[k] = True
        try:
            variance = self.concatenate('variance')
            weights = np.sqrt(at.divide0(1., variance))
        except TypeError:  # avoids later checks for None
            weights = np.ones_like(m_init.waves, dtype=np.float32)

        params = {k: v[0] for k, v in self.fit_params.items()}
        # This is logic for running non-interactively
        if sigma_clipping is None:
            sigma_clipping = params["niter"] > 0

        # We only send the unmasked points to the fitter, rather than sending a
        # MaskedArray and letting the fitter sort it out. This is because we
        # don't want to evaluate the fit at masked pixels, in case the continuum
        # is blowing up, which is what will happen otherwise since the cython
        # code to determine where to do the evaluation looks at the "x" (i.e.,
        # wavelength) values and doesn't know about the mask on "y".
        #weights = np.full_like(weights, 100)  # hack for now
        start_time = datetime.now()
        if sigma_clipping:
            sigma_clip_params = {k: v for k, v in params.items()
                                 if k in ('grow', 'sigma_lower', 'sigma_upper', 'niter')}
            fit_it = fitting.FittingWithOutlierRemoval(fitting.TRFLSQFitter(),
                                                       sigma_clip, maxiters=1,
                                                       **sigma_clip_params)
            m_final, new_mask = fit_it(m_init, m_init.waves[~mask], data[~mask],
                                       weights=weights[~mask], maxiter=10000)
        else:
            fit_it = fitting.LevMarLSQFitter()
            m_final = fit_it(m_init, m_init.waves[~mask], data[~mask],
                             weights=weights[~mask], maxiter=10000)
            new_mask = np.zeros_like(m_init.waves, dtype=bool)
        print(datetime.now() - start_time, "FINISHED FIT")

        # Reset masks to their original values
        for tspek, orig_mask in zip(self.spectra, original_masks):
            tspek.nddata.mask = orig_mask

        m_final.update_individual_models()
        return m_final, new_mask

    # Methods above should be common to all classes
    # Methods below are specific to this class
    def spectrum(self, index):
        return self.spectra[index]

    def concatenate(self, property='data'):
        return np.asarray([getattr(tspek, property)
                           for tspek in self.spectra]).flatten()

    def create_stellar_spectrum(self, wavelength=None, fluxden=None,
                                bbtemp=None, in_vacuo=True):
        """
        Modify the template A0 spectrum by scaling to a different blackbody
        temperature and flux, and return this spectrum.

        Parameters
        ----------
        wavelength: Quantity
            fiducial wavelength at which flux density is normalized
        fluxden: Quantity
            flux density at the fiducial wavelength
        bbtemp: float
            Blackbody temperature (in K) of new spectrum
        in_vacuo: bool
            Should the returned spectrum have wavelengths in vacuo? (rather than air)

        Returns
        -------
        arrays of wavelength (in nm) and flux density (in W/m^2/nm) of new spectrum
        """
        a0w, a0f = A0Spectrum.spectrum(r=self.resolution, in_vacuo=in_vacuo)
        bb_telluric = Planck(temperature=bbtemp)(a0w)
        bb_vega = Planck(temperature=A0Spectrum.bbtemp)(a0w)
        f = a0f * bb_telluric / bb_vega
        fluxden_in_flam_units = fluxden.to(u.W / (u.m ** 2 * u.nm),
                                           equivalencies=u.spectral_density(wavelength))
        scaling = fluxden_in_flam_units / np.interp(wavelength.to(u.nm).value, a0w, a0f)
        return a0w, f * scaling


class TelluricCorrector(Calibrator):
    def __init__(self, telluric_spectra, ui_params=None, tellabs_data=None,
                 tell_int_splines=None):
        # Lots of checking that all the TelluricSpectra have similar PCA models
        assert all(isinstance(t, TelluricSpectrum) for t in telluric_spectra)
        self.spectra = telluric_spectra
        self.npixels = np.array([tspek.waves.size for tspek in self.spectra])

        # The pixel data of the absorption models
        self.tellabs_data = tellabs_data
        # And the integral splines, pre-convolved
        self.tell_int_splines = tell_int_splines

        # The absorption spectra being used: at the standard's airmass
        self.std_abs = [np.ones((npix,), dtype=np.float32)
                        for npix in self.npixels]
        # at the object's airmass
        self.abs_final = [np.ones((npix,), dtype=np.float32)
                        for npix in self.npixels]

        if ui_params is not None:
            self.set_fitting_params(ui_params)
            self.reconstruct_points()

    def __len__(self):
        return len(self.spectra)

    @property
    def x(self):
        return [tspek.nddata.wcs(np.arange(npix) + self.reinit_params["pixel_shift"])
                for tspek, npix in zip(self.spectra, self.npixels)]

    @property
    def y(self):
        return [tspek.data for tspek in self.spectra]

    def set_fitting_params(self, ui_params):
        """Override because we don't have fitting parameters"""
        self.reinit_params = {p: getattr(ui_params, p)
                              for p in ui_params.reinit_params}

    def reconstruct_points(self, *args, **kwargs):
        """We need to calculate the absorption model if we're resampling.
        If we're just changing the airmass, we should be able to do that in the UI"""
        for i, tspek in enumerate(self.spectra):
            if self.reinit_params["apply_model"]:
                # Calculate the absorption model by integrating the spline
                # between the wavelength limits of each pixel
                edges = at.calculate_pixel_edges(self.x[i])
                dw_out = abs(np.diff(edges))
                result = abs(np.diff(self.tell_int_splines[i](edges)))
                self.std_abs[i][:] = result / dw_out
            else:
                # Interpolate the data from the TELLABS extension
                pixels = np.arange(self.npixels[i])
                self.std_abs[i][:] = np.interp(
                    pixels + self.reinit_params["pixel_shift"],
                    pixels, self.tellabs_data[i], left=np.nan, right=np.nan)

    def perform_fit(self, index, sigma_clipping=None):
        """No fitting, this just divides each spectrum by the absorption model"""
        delta_airmass = (self.reinit_params["sci_airmass"] -
                         self.reinit_params["std_airmass"])
        trans = self.std_abs[index]
        pix_to_correct = np.logical_and(trans > 0, trans < 1)
        tau = -np.log(trans[pix_to_correct]) / self.reinit_params["std_airmass"]
        trans[pix_to_correct] *= np.exp(-tau * delta_airmass)
        self.abs_final[index][:] = trans
        corrected_data = models.Tabular1D(
            points=self.x[index], lookup_table=at.divide0(self.spectra[index].data, trans))
        return corrected_data

    # def perform_all_fits(self):
    #     # This needs to return a model that describes the "fit" for each
    #     # spectrum; this needs to be a Tabular1D. But what we want the
    #     # Calibrator to return is the absorption model so let's store that
    #     # as well
    #     delta_airmass = (self.reinit_params["sci_airmass"] -
    #                      self.reinit_params["std_airmass"])
    #     for i, trans in enumerate(self.std_abs):
    #         pix_to_correct = np.logical_and(trans > 0, trans < 1)
    #         tau = -np.log(trans[pix_to_correct]) / self.reinit_params["std_airmass"]
    #         trans[pix_to_correct] *= np.exp(-tau * delta_airmass)
    #         self.abs_final[i][:] = trans
    #
    #     corrected_data = [models.Tabular1D(points=x, lookup_table=at.divide0(tspek.data, model))
    #                       for x, tspek, model in zip(self.x, self.spectra, self.abs_final)]
    #     return corrected_data
