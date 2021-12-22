import astrodata, gemini_instruments
import astrodata.testing
import pytest

from geminidr.gemini.lookups.keyword_comments import keyword_comments
from geminidr.gmos.primitives_gmos_image import GMOSImage
from gempy.gemini import gemini_tools as gt
from gempy.utils import logutils
from astropy.table import Table


test_data = [
    # (Filename, FWHM)
    ('N20180118S0344.fits', 1.32),
]


caltype_data = [
    ('arc', 'Arc spectrum'),
    ('bias', 'Bias Frame'),
    ('dark', 'Dark Frame'),
    ('fringe', 'Fringe Frame'),
    ('sky', 'Sky Frame'),
    ('flat', 'Flat Frame'),
]


@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("caltype, obj", caltype_data)
def test_convert_to_cal_header(caltype, obj, change_working_dir):
    """Check that header keywords have been updated and
    """
    # A random NIRI image
    ad = astrodata.open(astrodata.testing.download_from_archive('N20200127S0023.fits'))
    ad_out = gt.convert_to_cal_header(ad, caltype=caltype, keyword_comments=keyword_comments)

    # FITS WCS keywords only get changed at write-time, so we need to
    # write the file to disk and read it back in to confirm.
    with change_working_dir():
        ad_out.write("temp.fits", overwrite=True)
        ad = astrodata.open("temp.fits")

        assert ad.observation_type() == caltype.upper()
        # Let's not worry about upper/lowercase
        assert ad.object().upper() == obj.upper()

        assert ad.phu.get('RA', 0.) == ad.phu.get('DEC', 0.0) == 0.0

        assert ad.ra() == ad.dec() == 0.0


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


class TestCheckInputsMatch:

    def test_inputs_match(self, astrofaker):

        ad1 = astrofaker.create("GMOS-S", mode="IMAGE")
        ad1.init_default_extensions()

        ad2 = astrofaker.create("GMOS-S", mode="IMAGE")
        ad2.init_default_extensions()

        gt.check_inputs_match(ad1, ad2)

    def test_inputs_match_different_shapes(self, astrofaker):

        ad1 = astrofaker.create("GMOS-S", mode="IMAGE")
        ad1.init_default_extensions()
        for ext in ad1:
            ext.data = ext.data[20:-20, 20:-20]

        ad2 = astrofaker.create("GMOS-S", mode="IMAGE")
        ad2.init_default_extensions()

        with pytest.raises(ValueError):
            gt.check_inputs_match(ad1, ad2)

        gt.check_inputs_match(ad1, ad2, check_shape=False)
