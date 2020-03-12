import astrodata
import gemini_instruments
from geminidr.gmos.primitives_gmos_image import GMOSImage
from gempy.gemini import gemini_tools as gt
from astropy.table import Table
import os


def test_fit_continuum_slit_image(path_to_inputs):
    results = {'N20180118S0344': 1.32}
    for fname, fwhm in results.items():
        ad = astrodata.open(os.path.join(path_to_inputs, 'gt/slit_images',
                                         '{}.fits'.format(fname)))
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
