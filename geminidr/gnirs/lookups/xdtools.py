# Functions with a common API that can be imported and used by
# primitives in the CrossDispersed class

from gemini_instruments.gnirs import lookup


def order_info(ad, spec_order):
    """
    Returns the central wavelength and dispersion values (in nm) for a
    given spectral order of a GNIRS XD spectrum.

    Parameters
    ----------
    ad: AstroData
        the AstroData object under consideration
    spec_order: int
        the spectral order of interest in the cross-dispersed data

    Returns
    -------
    dict: {'cenwave': central wavelength in nm,
           'dispersion': dispersion in nm}
    """
    grating = ad._grating(pretty=True, stripID=True)
    camera = 'Short' if 'Short' in ad.camera() \
        else 'Long' if 'Long' in ad.camera() else None
    config = lookup.dispersion_by_config.get((grating, camera), {})
    filter_name = lookup.xd_orders.get(spec_order)
    dispersion = config.get(filter_name)
    cenwave = ad._grating_order() * ad.central_wavelength(asNanometers=True) / spec_order
    return {'cenwave': cenwave, 'dispersion': dispersion}
