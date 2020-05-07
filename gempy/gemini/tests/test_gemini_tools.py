import astrodata
import gemini_instruments
from geminidr.gmos.primitives_gmos_image import GMOSImage
from gempy.gemini import gemini_tools as gt
from astropy.table import Table

from gempy.testing import setup_log


def test_fit_continuum_slit_image(cache_file_from_archive):
    results = {'N20180118S0344.fits': 1.32}

    for fname, fwhm in results.items():
        ad = astrodata.open(cache_file_from_archive(fname))
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
