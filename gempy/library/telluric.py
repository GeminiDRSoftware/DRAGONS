# Things to do with fitting telluric absorption to real spectra
import os
from copy import deepcopy
import itertools

import numpy as np

from astropy.modeling import fitting
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy import units as u

from specutils.utils.wcs_utils import vac_to_air

from matplotlib import pyplot as plt

from geminidr.gemini.lookups import telluric as tel_lookups
from gempy.library import astrotools as at
from gempy.utils import logutils

from . import convolution
from .telluric_models import ArrayInterpolator, PCA, Planck

from datetime import datetime


PATH = os.path.split(tel_lookups.__file__)[0]


log = logutils.get_logger(__name__)


class TelluricModels:
    """A class to store the full, unconvolved telluric models in a way that
    only requires them to be read from disk the first time they are needed"""
    _wavelengths = None
    _models = None
    in_vacuo = True
    pca_file = None
    latest_filename = "pca_components_v1.0"
    #latest_filename = "pca_components_20240612"

    @classmethod
    def data(cls, pca_file=None, in_vacuo=True):
        """
        Returns the principal component models, reading them from a file if
        they haven't already been read. These are converted to air wavelengths
        if needed.

        The attributes in_vacuo and pca_file refer to the state of the stored
        models. The file only has to be read again if this method is called
        with values that differ from those stored.

        Returns
        -------
        two arrays:
        w: (N,) wavelengths of the models
        m: (M, N) data from the M principal components
        """
        if pca_file is None:
            pca_file = cls.latest_filename
        if (cls._wavelengths is None or cls._models is None or
                in_vacuo != cls.in_vacuo or pca_file != cls.pca_file):
            telluric_file = os.path.join(PATH, f'{pca_file}.fits')
            models = Table.read(telluric_file)
            waves = models['wavelength'].data
            if not in_vacuo:
                waves = vac_to_air(waves * u.nm).value
            cls._wavelengths = waves
            cls._models = np.empty((len(models.columns) - 1, len(models)))
            cls._models[0] = models['mean'].data
            for i in range(cls._models.shape[0] - 1):
                cls._models[i + 1] = models[f'comp{i:02d}'].data

            # Update a couple more attributes
            cls.in_vacuo = in_vacuo
            cls.pca_file = pca_file
        return cls._wavelengths, cls._models


class A0Spectrum:
    """Holds the Allard theoretical A0V spectrum (in F_lambda units)
    at requested spectral samplings, converted to air wavelengths if needed"""
    _dict = {}
    _raw_resolution = None
    bbtemp = 9650

    @classmethod
    def spectrum(cls, r=None, in_vacuo=True):
        if not cls._dict:
            phoenix_file = os.path.join(PATH, 'phoenix9600.fits')
            phoenix = Table.read(phoenix_file)
            waves = (phoenix['wavelength'].to(u.nm)).value
            if not in_vacuo:
                waves = vac_to_air(waves * u.nm).value
            # Only store the useful wavelength range. We need to resample
            # to a log scale since convolution, etc. assumes a constant
            # pixel scale across each output pixel and the raw A0 file has
            # jumps in sampling rate every 500nm.
            indices = np.logical_and(waves > 300, waves < 6000)
            wtemp = np.geomspace(300, 6000, indices.size, endpoint=True)
            cls._raw_resolution = wtemp[1] / (wtemp[1] - wtemp[0])
            log_interp_flux = np.interp(wtemp, waves[indices],
                                        phoenix['lte09600-4.00-0.0'].data[indices])
            cls._dict[None] = (wtemp, log_interp_flux)

        if r is None or cls._raw_resolution is None or r > cls._raw_resolution:
            return cls._dict[None]

        if r not in cls._dict:
            wtemp = np.exp(np.arange(*np.log([350, 5500, 1.+1./r])))
            cls._dict[r] = (wtemp, convolution.resample(wtemp, *cls._dict[None]))

        return cls._dict[r]


class TelluricSpectrum:
    """
    A class for holding information about an observed spectrum and an
    attempted telluric fit.
    """
    def __init__(self, ndd, line_spread_function=None, name=None,
                 in_vacuo=True, **lsf_kwargs):
        """

        Parameters
        ----------
        ndd: NDAstroData
            an object holding the data, variance, and mask
        line_spread_function: LineSpreadFunction instance
            object to handle convolution and resampling
        name: str
            an optional name for this spectrum
        in_vacuo: bool
            is the WCS a representation of vacuum (rather than air)
            wavelengths?
        lsf_kwargs: dict of {param_name: tuple}
            parameters for constructing a grid of convolved PCA modes
        """
        if len(ndd.shape) != 1:
            raise ValueError("TelluricSpectrum must be 1D")
        self.nddata = deepcopy(ndd)  # to be safe, it's not very large
        # Make our lives easier by not having to check "if mask is None"
        if ndd.mask is None:
            self.nddata.mask = np.zeros_like(ndd.data, dtype=bool)
        self.waves = ndd.wcs(np.arange(ndd.shape[0]))
        # dwaves is only used as the width of a pixel so should be +ve
        self.dwaves = abs(np.diff(at.calculate_pixel_edges(self.waves)))
        self.absolute_dispersion = np.median(self.dwaves)

        # Convert units to nm to make life easier
        wave_unit = ndd.wcs.output_frame.unit[0]
        if wave_unit != u.nm:
            try:
                scaling = (1 * wave_unit).to(u.nm).value
            except u.UnitConversionError:
                log.warning("Cannot convert wavelength units; assuming nm")
            else:
                self.waves *= scaling
                self.dwaves *= scaling

        self._domain = (self.waves[0], self.waves[-1])
        self.lsf = line_spread_function
        self.name = name
        self.in_vacuo = in_vacuo

        # Avoids having to do lots of "if self.intrinsic spectrum is not None"
        self.intrinsic_spectrum = np.ones_like(ndd.data)

    @property
    def data(self):
        return self.nddata.data

    @property
    def mask(self):
        return self.nddata.mask

    @property
    def variance(self):
        return self.nddata.variance

    def downsample_if_necessary(self, w, t):
        """
        Block-resample a spectrum or spectra if the resolution is much
        higher than that of the science spectrum. This reduces the time
        taken to perform the convolutions.
        """
        min_dw_data = abs(np.diff(self.waves)).min()
        min_dw_models = abs(np.diff(w[np.logical_and(w > self.waves.min(),
                                                     w < self.waves.max())])).min()
        while min_dw_data > min_dw_models * 10:
            w = w[1:-1:3]
            t = t[..., :w.size * 3].reshape(*t.shape[:-1], w.size, 3).mean(axis=-1)
            min_dw_models *= 3
        return w, t

    def set_pca(self, pca_file=None, lsf_params=None):
        """
        Set the PCA model for the telluric absorption. This is a convenience
        method for the case where the PCA model was not set at instantiation
        (e.g., because the convolution parameters were not known at that time).

        Parameters
        ----------
        lsf_params: dict
            parameters for the LSF convolution method
        """
        w_pca, t_pca = self.downsample_if_necessary(
            *TelluricModels.data(in_vacuo=self.in_vacuo, pca_file=pca_file))
        if lsf_params:
            # We only need an ArrayInterpolator if some of the params have
            # multiple values. Otherwise we might be constructing a model for
            # telluricCorrect() from single parameter values, for example.
            fixed_params, interp_params = {}, {}
            for k, v in lsf_params.items():
                try:
                    if len(v) == 1:
                        fixed_params[k] = v[0]
                    else:
                        interp_params[k] = v
                except TypeError:
                    fixed_params[k] = v
            # print("FIXED ", fixed_params)
            # print("INTERP", interp_params)
            if interp_params:
                pca_data = np.stack([self.lsf.convolve_and_resample(
                    self.waves, w_pca, t_pca, **fixed_params,
                    **dict(zip(interp_params.keys(), params)))
                    for params in itertools.product(*interp_params.values())])
                self.pca = PCA(ArrayInterpolator(interp_params.values(), pca_data),
                               name=TelluricModels.pca_file)
            else:
                convolved_models = self.lsf.convolve_and_resample(
                    self.waves, w_pca, t_pca, **fixed_params)
                self.pca = PCA(convolved_models, name=TelluricModels.pca_file)
        else:
            convolved_models = self.lsf.convolve_and_resample(
                self.waves, w_pca, t_pca)
            self.pca = PCA(convolved_models, name=TelluricModels.pca_file)

    def set_intrinsic_spectrum(self, w, data, **lsf_kwargs):
        """
        Set the intrinsic spectrum of the object being used to determine the
        telluric absorption (probably a star). A spectrum is provided at
        arbitrarily high resolution and is convolved with the Line Spread
        Function and resampled to the same wavelength array as the data.

        Parameters
        ----------
        w: ndarray
            wavelengths of the data
        data: ndarray
            data to be convolved
        lsf_kwargs: dict
            parameters to be passed to the LSF convolution method
        """
        int_spectrum = self.lsf.convolve_and_resample(
            self.waves, *self.downsample_if_necessary(w, data),
            **lsf_kwargs)
        if hasattr(data, 'unit'):
            self.intrinsic_spectrum = int_spectrum * data.unit
        else:
            self.intrinsic_spectrum = int_spectrum

    def make_stellar_mask(self, threshold=0.9, grow=1, max_contiguous=40,
                          r=None, plot=False):
        """
        Identify regions of stellar absorption. This needs to be a method
        because it needs access to the convolution kernel.

        This method fits a Planck function to the convolved model A0 spectrum
        and flags points that are lower than the fit. These are typically
        stellar absorption features, but the Paschen and Brackett limits also
        get flagged and we probably don't want that.

        This mask is *not* stored as an attribute because it is not handled by
        the Model classes that use TelluricSpectrum -- they only use the mask
        in the NDData object, which must be combined with the stellar mask if
        desired. Since this method has to perform a convolution, it should
        ideally not be called more than once; store the result if required.

        Parameters
        ----------
        threshold: float (0-1)
            points below this threshold are flagged
        grow: int
            growth radius for bad points
        max_contiguous: float
            groups of flagged points (before growth) larger than this
            wavelength range (in nm) get unflagged
        r: float
            spectral resolution (sampling) of A0 model to use

        Returns
        -------
        boolean array of masked points
        """
        start = datetime.now()
        stellar_spectrum = self.lsf.convolve_and_resample(self.waves,
                                                          *A0Spectrum.spectrum(r=r))
        #stellar_spectrum = self.convolve_and_resample(
        #    *A0Spectrum.spectrum(r=r), convolution_list=self.convolution_list)
        fit_it = fitting.FittingWithOutlierRemoval(fitting.LevMarLSQFitter(),
                                                   outlier_func=sigma_clip,
                                                   sigma_lower=1, sigma_upper=3,
                                                   maxiters=2)
        planck = Planck(temperature=A0Spectrum.bbtemp,
                        scale=np.median(stellar_spectrum))
        m_planck, _ = fit_it(planck, self.waves, stellar_spectrum)
        stellar_mask = stellar_spectrum / m_planck(self.waves) < threshold
        masked_slices = np.ma.clump_masked(np.ma.masked_array(
            self.waves, mask=stellar_mask))
        stellar_mask = np.zeros_like(stellar_mask)
        for _slice in masked_slices:
            if abs(self.waves[_slice.stop - 1] -
                   self.waves[_slice.start]) <= max_contiguous:
                stellar_mask[_slice] = True
        if grow > 0:
            stellar_mask = at.boxcar(stellar_mask, operation=np.logical_or,
                                     size=grow)

        # debugging
        if plot:
            fig, ax = plt.subplots()
            ax.plot(self.waves, stellar_spectrum, 'ko', label="A0 spectrum")
            ax.plot(self.waves[stellar_mask], stellar_spectrum[stellar_mask],
                    'ro', label="Masked")
            ax.plot(self.waves, m_planck(self.waves), 'b-', label="Planck fit")
            plt.show()

        return stellar_mask
