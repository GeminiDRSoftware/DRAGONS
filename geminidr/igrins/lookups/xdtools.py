# Functions with a common API that can be imported and used by
# primitives in the CrossDispersed class

def order_info(ad, spec_order):
    """
    Returns the central wavelength and dispersion values (in nm) for a
    given spectral order of an IGRINS-2 XD spectrum.

    These values are approximate and this code may not be used in the
    final version of the recipe.

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
    cenwave = 178000. / spec_order
    dispersion = 0.00118 / spec_order
    return {'cenwave': cenwave, 'dispersion': dispersion}
