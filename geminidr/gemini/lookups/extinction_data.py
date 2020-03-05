import numpy as np
from astropy import units as u
from scipy.interpolate import InterpolatedUnivariateSpline as spline

# Mapping from telescope name to site's extinction curve
# We have no curve for CPO, so use the MKO one for now
telescope_sites = {'Gemini-North': 'MKO',
                   'Gemini-South': 'MKO'}

# Extinction curves for supported sites, as a function of wavelength in nm
# Units are mag/airmass
extinction_curves = {
    # From Buton et al. (2013, A&A 549, A8), 310nm point from Gemini website
    'MKO': spline(np.arange(310, 1001, 10),
                  (1.37, 0.856, 0.588, 0.514, 0.448, 0.400, 0.359, 0.323, 0.292,
                   0.265, 0.241, 0.220, 0.202, 0.185, 0.171, 0.159, 0.147, 0.139,
                   0.130, 0.125, 0.119, 0.114, 0.113, 0.109, 0.106, 0.107, 0.108,
                   0.103, 0.098, 0.098, 0.092, 0.084, 0.078, 0.070, 0.065, 0.060,
                   0.056, 0.052, 0.048, 0.044, 0.042, 0.039, 0.037, 0.035, 0.033,
                   0.032, 0.030, 0.029, 0.028, 0.027, 0.026, 0.025, 0.024, 0.023,
                   0.023, 0.022, 0.021, 0.021, 0.020, 0.019, 0.019, 0.018, 0.018,
                   0.017, 0.017, 0.016, 0.016, 0.015, 0.015, 0.014))
}


def extinction(wave, site=None, telescope=None):
    """
    This function returns the extinction (in mag/airmass) at the specified
    input wavelengths. Wavelengths should be Quantity objects but, if they
    are scalars, units of nm are assumed.

    Parameters
    ----------
    wave: float/array/Quantity
        wavelength(s) at which to derive extinction (if no units, the
        wavelength(s) are assumed to be in nm)
    site: str/None
        name of site (key for extinction curve)
    telescope: str/None
        name of telescope (maps to site)

    Returns
    -------
    arrray: extinction in magnitudes/airmass at requested wavelength(s)
    """
    if telescope in telescope_sites:
        site = telescope_sites[telescope]
    elif site not in extinction_curves:
        raise KeyError("Site {} not recongized".format(site))
    try:
        wave_in_nm = wave.to(u.nm)
    except AttributeError:
        # Assume it's nm already
        wave_in_nm = wave
    return extinction_curves[site](wave_in_nm)