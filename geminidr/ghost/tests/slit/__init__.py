import pytest

import datetime

from astropy.io import fits

import astrodata, gemini_instruments
from geminidr.ghost.polyfit.slitview import SlitView
from geminidr.ghost.lookups import polyfit_lookup


NO_SLITS = 10
EXPTIME_SLITS = 10.
SLIT_UT_START = datetime.datetime(2023, 3, 1, 0, 0)
STRFTIME = '%H:%M:%S.%f'
STRFDATE = '%Y-%m-%d'


@pytest.fixture
def ad_slit():
    """
    Generate a package of dummy slit files.

    .. note::
        Fixture.
    """
    rawfilename = 'testslitpackage.fits'

    # Create the AstroData object
    phu = fits.PrimaryHDU()
    phu.header.set('CAMERA', 'slit')
    phu.header.set('CCDNAME', 'Sony-ICX674')
    phu.header.set('DATE-OBS', SLIT_UT_START.strftime(STRFDATE))
    phu.header.set('UTSTART', SLIT_UT_START.strftime(STRFTIME))
    phu.header.set('UTEND', (SLIT_UT_START + datetime.timedelta(
        seconds=(NO_SLITS + 1) * EXPTIME_SLITS)).strftime(STRFTIME))
    phu.header.set('INSTRUME', 'GHOST')
    phu.header.set('DATALAB', 'test')
    phu.header.set('SMPNAME', 'LO_ONLY')

    hdus = []
    for i in range(NO_SLITS):
        # Dummy data plane for now
        hdu = fits.ImageHDU(data=[0], name='SCI')
        hdu.header.set('CAMERA', phu.header.get('CAMERA'))
        hdu.header.set('CCDNAME', phu.header.get('CCDNAME'))
        hdu.header.set('EXPID', i + 1)
        hdu.header.set('CCDSUM', '2 2')
        hdu.header.set('EXPUTST', (SLIT_UT_START +
                                   datetime.timedelta(
                                       seconds=(i * 0.2) * EXPTIME_SLITS
                                   )).strftime(STRFTIME))
        hdu.header.set('EXPUTST', (SLIT_UT_START +
                                   datetime.timedelta(
                                       seconds=((i * 0.2) + 1) * EXPTIME_SLITS
                                   )).strftime(STRFTIME))
        hdu.header.set('GAIN', 1.0)
        hdu.header.set('RDNOISE', 8.0)
        hdus.append(hdu)

    # Create AstroData
    ad = astrodata.create(phu, hdus)
    ad.filename = rawfilename

    # We need to have a decent-looking slitview image in order to
    # scale by fluxes
    slitv_fn = polyfit_lookup.get_polyfit_filename(
        None, 'slitv', 'std', ad.ut_date(), ad.filename, 'slitvmod')
    slitvpars = astrodata.open(slitv_fn)
    sview = SlitView(None, None, slitvpars.TABLE[0], mode=ad.res_mode())
    slit_data = sview.fake_slitimage(seeing=0.7)
    for ext in ad:
        ext.data = slit_data.copy()

    return ad
