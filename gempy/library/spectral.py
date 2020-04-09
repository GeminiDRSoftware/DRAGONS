# Copyright(c) 2019-2020 Association of Universities for Research in Astronomy, Inc.
#
from astrodata import AstroData, NDAstroData
from geminidr.gemini.lookups import DQ_definitions as DQ

from specutils import Spectrum1D, SpectralRegion
from astropy import units as u
from astropy.nddata import NDData
import numpy as np
from astropy.modeling import models
from gwcs import wcs as gWCS
from gwcs import coordinate_frames as cf

from . import astromodels as am

class Spek1D(Spectrum1D, NDAstroData):
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
    def __init__(self, spectrum=None, spectral_axis=None, wcs=None, **kwargs):
        # This handles cases where arithmetic is being performed, and an
        # object is created that's just a number
        if not isinstance(spectrum, (AstroData, NDData)):
            super().__init__(spectrum, spectral_axis=spectral_axis, wcs=wcs, **kwargs)
            return

        if isinstance(spectrum, AstroData) and not spectrum.is_single:
            raise TypeError("Input spectrum must be a single AstroData slice")

        # Unit handling
        try:  # for NDData-like
            flux_unit = spectrum.unit
        except AttributeError:
            try:  # for AstroData
                flux_unit = u.Unit(spectrum.hdr.get('BUNIT'))
            except (TypeError, ValueError):  # unknown/missing
                flux_unit = None
        if flux_unit is None:
            flux_unit = u.dimensionless_unscaled
        try:
            kwargs['mask'] = spectrum.mask
        except AttributeError:
            flux = spectrum
        else:
            flux = spectrum.data
            kwargs['uncertainty'] = spectrum.uncertainty

        # If spectrum was a Quantity, it already has units so we'd better
        # not multiply them in again!
        if not isinstance(flux, u.Quantity):
            flux *= flux_unit

        # If no wavelength information is included, get it from the input
        if spectral_axis is None and wcs is None:
            if isinstance(spectrum, AstroData):
                if spectrum.nddata.wcs is not None:
                    wcs = spectrum.nddata.wcs
                else:
                    spec_unit = u.Unit(spectrum.hdr.get('CUNIT1', 'nm'))
                    try:
                        wavecal = dict(zip(spectrum.WAVECAL["name"],
                                           spectrum.WAVECAL["coefficients"]))
                    except (AttributeError, KeyError):  # make a Model from the FITS WCS info
                        det2wave = (models.Shift(1-spectrum.hdr['CRPIX1']) |
                                    models.Scale(spectrum.hdr['CD1_1']) |
                                    models.Shift(spectrum.hdr['CRVAL1']))
                    else:
                        det2wave = am.dict_to_chebyshev(wavecal)
                        det2wave.inverse = am.make_inverse_chebyshev1d(det2wave, sampling=1)
                        spec_unit = u.nm
                    detector_frame = cf.CoordinateFrame(1, axes_type='SPATIAL',
                                    axes_order=(0,), unit=u.pix, axes_names='x')
                    spec_frame = cf.SpectralFrame(unit=spec_unit, name='lambda')
                    wcs = gWCS.WCS([(detector_frame, det2wave),
                                    (spec_frame, None)])
            else:
                wcs = spectrum.wcs  # from an NDData-like object

        super().__init__(flux=flux, spectral_axis=spectral_axis, wcs=wcs, **kwargs)
        self.filename = getattr(spectrum, 'filename', None)

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
        x1, x2: float, float
            pixel locations of the start end end of the spectral region
        """
        limits = []
        for loc in subregion:
            if loc.unit.is_equivalent(u.pix):
                loc_in_pix = loc.value
            else:
                # Convert to the correct units
                wave = loc.to(self.spectral_axis.unit, u.spectral())
                loc_in_pix = self.wcs.world_to_pixel([wave])
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

    def signal(self, region):
        """
        Measure the signal over a defined region of the spectrum. If the
        units of the spectrum are a flux density, then a flux is returned,
        otherwise the corresponding pixel values are simply summed.

        Parameters
        ----------
        region: :class:`~SpectralRegion`
            the region of the spectrum over which the signal is calculated

        Returns
        -------
        3-tuple: flux, mask, and variance summed over selected region(s)
        """
        flux_density = (self.unit * self.spectral_axis.unit).is_equivalent(u.Jy * u.Hz)
        flux = 0
        mask = DQ.good
        variance = None if self.variance is None else 0

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

            # Construct a "Manhattan skyline"
            ix1 = int(np.floor(limits[0] + 0.5))
            ix2 = int(np.floor(limits[1] + 0.5))

            # We need to cope with limits[1]==len(flux)-0.5, i.e.,
            # at the right-hand edge of the spectrum
            edges = np.repeat(np.arange(ix1, min(ix2, len(self.flux)-1)), 2) + 0.5
            x = np.insert(edges, 0, limits[0])
            x = np.append(x, limits[1])
            if flux_density:
                x = self.wcs.pixel_to_world(x)

            flux += np.trapz(np.repeat(self.flux[ix1:ix2+1], 2), x)
            if self.mask is not None:
                mask |= np.bitwise_or.reduce(self.mask[ix1:ix2+1])
            if self.variance is not None:
                # TODO: May need to come back to this in case variance gets units
                y = np.repeat(self.variance[ix1:ix2+1], 2) * self.unit * self.unit
                # The variance of each pixel needs to be multiplied by
                # the wavelength coverage of that (partial pixel)
                if flux_density:
                    dlambda = np.diff(x)[::2]
                    y = y * np.repeat(dlambda, 2)
                variance += np.trapz(y, x)

        return flux, mask, variance
