# Functions with a common API that can be imported and used by
# primitives in the CrossDispersed class

from astropy.modeling import models

from gemini_instruments.gnirs import lookup


def initial_wave_models(ad):
    """
    Yields initial wavelength solution models for each extension in the input

    Parameters
    ----------
    ad: AstroData
        the AstroData object under consideration

    Yields
    ------
        a Chebyshev1D model describing the wavelength solution for each
        extension in the input AstroData
    """
    grating = ad._grating(pretty=True, stripID=True)
    camera = 'Short' if 'Short' in ad.camera() \
        else 'Long' if 'Long' in ad.camera() else None
    config = lookup.dispersion_by_config.get((grating, camera), {})

    for ext in ad:
        spec_order = ext.SLITEDGE["specorder"][0]
        filter_name = lookup.xd_orders.get(spec_order)
        dispersion = config.get(filter_name)
        cenwave = ad._grating_order() * ad.central_wavelength(asNanometers=True) / spec_order
        npix = ext.shape[0]
        # Return the model in this form so that it has a ready-made inverse
        yield models.Shift(-0.5*(npix-1)) | models.Scale(dispersion) | models.Shift(cenwave)
