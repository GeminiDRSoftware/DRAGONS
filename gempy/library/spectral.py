# Copyright(c) 2019-2025 Association of Universities for Research in Astronomy, Inc.
#
from astrodata import AstroData, AstroDataMixin
from geminidr.gemini.lookups import DQ_definitions as DQ

from specutils import Spectrum1D, SpectralRegion
from astropy import units as u
from astropy.coordinates import SpectralCoord
from astropy.nddata import NDData, NDDataRef
import numpy as np
from astropy.modeling import models
from gwcs import wcs as gWCS
from gwcs import coordinate_frames as cf

from . import astromodels as am


class Spek1D(AstroDataMixin, NDDataRef):
    """
    Spectrum container for 1D spectral data, utilizing benefits of
    AstroData. This enhances :class:`~specutils.Spectrum1D` by having
    the `variance` attribute and allowing instantiation from a
    :class:`~AstroData` object, with its pre-existing WCS information.

    Parameters
    ----------
    spectrum: :class:`~AstroData` or :class:`~NDData` or :class:`~astropy.units.Quantity`
        Object containing flux (and maybe mask and uncertainty) data
    spectral_axis: :class:`~astropy.units.Quantity`
        Dispersion information (if not included in `spectrum`)
    wcs: `astropy.wcs.WCS` or `gwcs.wcs.WCS`
        WCS information (if not included in `spectrum`)
    """
    __add__ = AstroData.__add__
    __sub__ = AstroData.__sub__
    __mul__ = AstroData.__mul__
    __truediv__ = AstroData.__truediv__
    __iadd__ = AstroData.__iadd__
    __isub__ = AstroData.__isub__
    __imul__ = AstroData.__imul__
    __itruediv__ = AstroData.__itruediv__
    __rmul__ = __mul__
    __rtruediv__ = AstroData.__rtruediv__

    # Cannot __radd__ or __rsub__ as the units appear to get lost

    add = __iadd__
    subtract = __isub__
    multiply = __imul__
    divide = __itruediv__

    def __init__(self, spectrum=None, spectral_axis=None, wcs=None, unit=None,
                 copy=False):
        if isinstance(spectrum, AstroData) and not spectrum.is_single:
            raise TypeError("Input spectrum must be a single AstroData slice")

        if isinstance(spectrum, (AstroData, NDData)):
            data = spectrum.data
            mask = spectrum.mask
            uncertainty = spectrum.uncertainty
            # Unit handling
            if unit is None:
                if getattr(spectrum, 'unit', None) is None:
                    try:  # for AstroData
                        unit = u.Unit(spectrum.hdr.get('BUNIT'))
                    except (TypeError, ValueError):  # unknown/missing
                        unit = None
                else:
                    unit = spectrum.unit
        else:
            if isinstance(spectrum, u.Quantity):
                if unit is None:
                    unit = spectrum.unit
                spectrum = spectrum.value
            try:  # MaskedArray
                mask = spectrum.mask
                data = spectrum.data
            except AttributeError:
                data = spectrum
        #if self.unit is None:
        #    self.unit = u.dimensionless_unscaled

        if len(data.shape) != 1:
            raise ValueError("Input spectrum must be one-dimensional")

        # If no wavelength information is included, get it from the input
        if spectral_axis is None and wcs is None:
            if isinstance(spectrum, AstroData):
                if spectrum.wcs is not None:
                    wcs = spectrum.wcs
                else:
                    spec_unit = u.Unit(spectrum.hdr.get('CUNIT1', 'nm'))
                    try:
                        det2wave = am.table_to_model(spectrum.WAVECAL)
                    except AttributeError:
                        # make a Model from the FITS WCS info
                        det2wave = (models.Shift(1 - spectrum.hdr['CRPIX1']) |
                                    models.Scale(spectrum.hdr['CD1_1']) |
                                    models.Shift(spectrum.hdr['CRVAL1']))
                    else:
                        det2wave.inverse = am.make_inverse_chebyshev1d(det2wave, sampling=1)
                        spec_unit = det2wave.meta["yunit"]
                    detector_frame = cf.CoordinateFrame(1, axes_type='SPATIAL',
                                    axes_order=(0,), unit=u.pix, axes_names='x')
                    spec_frame = cf.SpectralFrame(unit=spec_unit, name='lambda')
                    wcs = gWCS.WCS([(detector_frame, det2wave),
                                    (spec_frame, None)])
            else:
                wcs = spectrum.wcs  # from an NDData-like object

        super().__init__(data=data, mask=mask, uncertainty=uncertainty,
                         wcs=wcs, unit=unit, copy=copy)
        if spectral_axis is None:
            self.spectral_axis = (wcs(np.arange(0, data.size)) *
                                  self.wcs.output_frame.unit[0])
        else:
            self.spectral_axis = spectral_axis
        self.filename = getattr(spectrum, 'filename', None)

    @property
    def flux(self):
        if self.unit:
            return self.data * self.unit
        return self.data * u.dimensionless_unscaled

    @flux.setter
    def flux(self, value):
        if value is None:
            raise ValueError("Flux cannot be None")
        if not hasattr(value, 'shape') or value.shape != self.shape:
            raise ValueError("Flux must be an array of same shape as existing")
        if isinstance(value, u.Quantity):
            self._data = value.value
            self.unit = value.unit
        else:
            self._data = value

    def _standard_nddata_op(self, fn, operand):
        """Here to cope with our mask-handling, mainly"""
        result = fn(self, operand, handle_mask=np.bitwise_or, handle_meta='first_found')
        self._data = result.data
        self._mask = result.mask
        self._uncertainty = result.uncertainty
        self._unit = result.unit
        del result

    def _get_pixel_limits(self, subregion, constrain=True):
        """
        Calculate and return the left and right pixel locations defined
        by the lower and upper bounds of the subregion. Unlike the `specutils`
        function `_to_edge_pixel`, this returns floats, not ints.

        Parameters
        ----------
        subregion: 2-tuple
            the start and end locations of the spectral region
        constrain: bool
            prevent the results extending beyond the edges of the spectrum?

        Returns
        -------
        list
            Contains the pixel locations of the start end end of the spectral
            region.
        """
        limits = []
        for loc in subregion:
            if loc.unit.is_equivalent(u.pix):
                loc_in_pix = loc.value
            else:
                # Convert to the correct units
                wave = loc.to(self.spectral_axis.unit, u.spectral())
                loc_in_pix = self.wcs.world_to_pixel(SpectralCoord(wave))
            if constrain:
                loc_in_pix = min(max(loc_in_pix, -0.5), len(self.spectral_axis)-0.5)
            limits.append(float(loc_in_pix))
        return limits

    def pixel_sizes(self):
        """
        Provide the wavelength extent of each pixel in the spectrum.

        Returns
        -------
        Quantity: pixel extent
        """
        pixel_edges = np.arange(len(self.spectral_axis)+1) - 0.5
        wave_edges = self.wcs.pixel_to_world(pixel_edges)
        return np.diff(wave_edges)

    def signal(self, region, interpolate=DQ.good):
        """
        Measure the signal over a defined region of the spectrum. If the
        units of the spectrum are a flux density, then a flux is returned,
        otherwise the corresponding pixel values are simply summed.
        Optionally, certain mask bits will result in the pixel being ignored
        (effectively interpolated over).

        Parameters
        ----------
        region: :class:`~SpectralRegion`
            the region of the spectrum over which the signal is calculated
        interpolate: int
            pixels with these mask bits set will be ignored

        Returns
        -------
        flux : units.Quantity
            Integrated flux over the selected region(s).
        mask : numpy.uint16
            Mask over the selected region(s).
        variance : units.Quantity
            Variance over selected region(s).
        """
        flux_density = (self.unit * self.spectral_axis.unit).is_equivalent(u.Jy * u.Hz)
        flux = 0.
        mask = DQ.good
        variance = None if self.variance is None else 0.

        for subregion in region._subregions:
            # If the region extends beyond the spectrum, we flag with NO_DATA
            limits = sorted(self._get_pixel_limits(subregion, constrain=False))
            for i in (0, 1):
                if limits[i] < -0.5:
                    limits[i] = -0.5
                    mask |= DQ.no_data
                if limits[i] > len(self.spectral_axis) - 0.5:
                    limits[i] = len(self.spectral_axis) - 0.5
                    mask |= DQ.no_data

            # If region is entirely off-spectrum, move along
            if limits[0] == len(self.spectral_axis) - 0.5 or limits[1] == -0.5:
                continue

            ix1 = int(np.floor(limits[0] + 0.5))
            ix2 = int(np.floor(limits[1] + 0.5))

            # We need to cope with limits[1]==len(flux)-0.5, i.e.,
            # at the right-hand edge of the spectrum
            edges = np.r_[limits[0],
                          np.arange(ix1, min(ix2, self.size-1)) + 0.5,
                          limits[1]]
            x = self.wcs.pixel_to_world(edges) if flux_density else edges
            widths = np.diff(x)

            if self.mask is None:
                included_pixels = None
            else:
                included_pixels = (self.mask[ix1:ix2+1] & interpolate) == DQ.good
                mask |= np.bitwise_or.reduce(self.mask[ix1:ix2+1][included_pixels])
            if widths[included_pixels].sum() == 0:
                included_pixels = None  # avoid 0/0 division
            flux += ((widths * self.flux[ix1:ix2+1])[included_pixels].sum() *
                     widths.sum() / widths[included_pixels].sum())

            if self.variance is not None:
                # TODO: May need to come back to this in case variance gets units
                y = self.variance[ix1:ix2+1] * self.unit * self.unit
                # The variance of each pixel needs to be multiplied by
                # the wavelength coverage of that (partial pixel)
                if flux_density:
                    y *= widths
                variance += ((widths * y)[included_pixels].sum() *
                             widths.sum() / widths[included_pixels].sum())

        return flux, mask, variance

    def asSpectrum1D(self):
        """Create a :class:`~specutils.Spectrum1D` object of this spectrum"""
        kwargs = {}
        if self.wcs is None:
            kwargs['spectral_axis'] = self.spectral_axis
        return Spectrum1D(flux=self.flux, mask=self.mask, uncertainty=self.uncertainty,
                          wcs=self.wcs, **kwargs)