import astrodata
import astrodata.testing
import gemini_instruments
import pytest

from geminidr.gmos.primitives_gmos_image import GMOSImage
from gempy.gemini import gemini_tools as gt
from gempy.utils import logutils
from astropy.table import Table


test_data = [
    # (Filename, FWHM)
    ('N20180118S0344.fits', 1.32),
]


@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("fname, fwhm", test_data)
def test_fit_continuum_slit_image(fname, fwhm, change_working_dir):

    with change_working_dir():

        log_file = 'log_{}.log'.format(fname.replace('.fits', ''))
        logutils.config(file_name=log_file)

        ad = astrodata.open(astrodata.testing.download_from_archive(fname))
        p = GMOSImage([ad])
        p.prepare(attach_mdf=True)
        p.addDQ()
        p.trimOverscan()
        p.ADUToElectrons()
        p.mosaicDetectors()
        tbl = gt.fit_continuum(p.streams['main'][0])[0]

        assert isinstance(tbl, Table)
        assert len(tbl) == 1
        assert abs(tbl['fwhm_arcsec'].data[0] - fwhm) < 0.05
