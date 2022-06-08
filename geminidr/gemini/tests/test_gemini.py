import copy
import datetime
import pytest

from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord

import astrodata
from gempy.utils import logutils
from geminidr.gemini.primitives_gemini import Gemini

# --- Delete me? ---
# @pytest.fixture(scope='module')
# def ad(path_to_inputs):
#
#     return astrodata.open(
#         os.path.join(path_to_inputs, 'N20020829S0026.fits'))

# --- Delete me? ---
# acqimage = GmosAcquisition(ad, 'GN2018AQ903-01.fits', TESTDATAPATH)
# box = acqimage.get_mos_boxes()[0]
#   export PRIMITIVE_TESTDATA=/net/hbf-nfs/sci/rtfperm/dragons/testdata/GMOS
# # next three lines are needed to initialize variables needed to set up acqbox
# actual_area = ACQUISITION_BOX_SIZE * ACQUISITION_BOX_SIZE
# scidata = box.get_data()
# acqbox = find_optimal(acqimage, scidata, mos.measure_box,
#                       mos.get_box_error_func(actual_area,
#                                              box.unbinned_pixel_scale()))


STAR_POSITIONS = [(200., 200.), (300.5, 800.5)]


# --- Fixtures ---
@pytest.fixture()
def gemini_image(astrofaker):
    af = astrofaker.create('NIRI', 'IMAGE')
    af.init_default_extensions()
    # SExtractor struggles if the background is noiseless
    af.add_read_noise()
    for x, y in STAR_POSITIONS:
        af[0].add_star(amplitude=500, x=x, y=y)
    return af  # geminiimage([af])


@pytest.fixture(scope='function')
def niri_sequence():
    """Creates a 3x3 NIRI dither sequence but does not update the WCS;
    only the offsets show the dither"""
    import astrofaker
    adinputs = [astrofaker.create('NIRI', 'IMAGE', filename=f"N20010101S{i:04d}.fits") for i in range(1, 10)]
    for i, ad in enumerate(adinputs):
        # All ADs have the same WCS. Modify the offsets to be inconsistent
        raoff = (i // 3 - 1) * 5
        decoff = (i % 3 - 1) * 5
        ad.phu['RAOFFSET'] = raoff
        ad.phu['DECOFFSE'] = decoff
        ad.phu['POFFSET'] = raoff
        ad.phu['QOFFSET'] = decoff
        ad.phu['DATE-OBS'] = '2000-01-01'
        ad.phu['UT'] = (datetime.datetime(year=2001, month=1, day=1) +
                        datetime.timedelta(seconds=i*(ad.exposure_time()+5))).time().isoformat()
        ad.init_default_extensions()
    return adinputs

#@pytest.fixture(scope='module', autouse=True)
#def setup_log(change_working_dir):
#    with change_working_dir():
#        logutils.config(file_name='test_gemini.log')


# --- Tests ---
def test_standardize_observatory_headers(gemini_image):

    test_gemini = Gemini([gemini_image])
    processed_af = test_gemini.standardizeObservatoryHeaders()[0]
    expected_timestamp = processed_af.phu['SDZHDRSG']

    assert isinstance(expected_timestamp, str), "phu SDZHDRSG tag not found!"

    new_af = copy.deepcopy(processed_af)
    test_gemini2 = Gemini([new_af])
    processed_af2 = test_gemini2.standardizeObservatoryHeaders()[0]
    expected_timestamp2 = processed_af2.phu['SDZHDRSG']

    assert (expected_timestamp == expected_timestamp2)


def test_standardize_wcs_not_offsetting_fail(niri_sequence):
    """Confirm that the reduction throws a ValueError by deafult"""
    p = Gemini(niri_sequence)
    with pytest.raises(ValueError):
        p.standardizeWCS()


@pytest.mark.parametrize("bad_wcs", ("ignore", "fix", "new"))
def test_standardize_wcs_handle(bad_wcs, niri_sequence):
    """Confirm that the reduction can create the correct offsets"""
    niri_sequence[0].phu['RA'] = 270
    p = Gemini(niri_sequence)
    p.standardizeWCS(bad_wcs=bad_wcs)
    coords = [SkyCoord(*ad[0].wcs(512, 512), unit='deg') for ad in p.streams['main']]
    offsets = [coords[4].separation(c).arcsec for c in coords]
    if bad_wcs == "ignore":
        assert_allclose(offsets, 0, atol=1e-3)
    else:
        del offsets[4]
        assert all([4.9 < offset < 7.1 for offset in offsets])
        sep_from_target = coords[4].separation(SkyCoord(ra=270, dec=0, unit='deg')).arcsec
        if bad_wcs == "fix":
            # Doesn't update RA and DEC to PHU values
            assert sep_from_target > 30000
        else:
            # Does update them, so the pointing will all be around RA=270
            assert sep_from_target < 2
