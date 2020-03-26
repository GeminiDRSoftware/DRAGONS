from gempy.library.astromodels import dict_to_chebyshev
from numpy.testing import assert_allclose


def assert_have_same_distortion(ad, ad_ref):
    """
    Checks if two :class:`~astrodata.AstroData` (or any subclass) have the
    same distortion.

    Parameters
    ----------
    ad : :class:`astrodata.AstroData`
        AstroData object to be checked.
    ad_ref : :class:`astrodata.AstroData`
        AstroData object used as reference

    """
    for ext, ext_ref in zip(ad, ad_ref):
        distortion = dict(zip(ext.FITCOORD["name"],
                              ext.FITCOORD["coefficients"]))
        distortion = dict_to_chebyshev(distortion)

        distortion_ref = dict(zip(ext_ref.FITCOORD["name"],
                                  ext_ref.FITCOORD["coefficients"]))
        distortion_ref = dict_to_chebyshev(distortion_ref)

        assert isinstance(distortion, type(distortion_ref))
        assert_allclose(distortion.parameters, distortion_ref.parameters)


def assert_wavelength_solutions_are_close(ad, ad_ref):
    """
    Checks if two :class:`~astrodata.AstroData` (or any subclass) have the
    wavelength solution.

    Parameters
    ----------
    ad : :class:`astrodata.AstroData` or any subclass
        AstroData object to be checked.
    ad_ref : :class:`astrodata.AstroData` or any subclass
        AstroData object used as reference

    """
    for ext, ext_ref in zip(ad, ad_ref):
        assert hasattr(ext, "WAVECAL")
        wcal = dict(zip(ext.WAVECAL["name"], ext.WAVECAL["coefficients"]))
        wcal = dict_to_chebyshev(wcal)

        assert hasattr(ext_ref, "WAVECAL")
        wcal_ref = dict(zip(ad[0].WAVECAL["name"],
                            ad[0].WAVECAL["coefficients"]))
        wcal_ref = dict_to_chebyshev(wcal_ref)

        assert isinstance(wcal, type(wcal_ref))
        assert_allclose(wcal.parameters, wcal_ref.parameters)
