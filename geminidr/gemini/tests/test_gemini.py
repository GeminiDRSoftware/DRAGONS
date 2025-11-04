import copy
import datetime
import pytest

import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord

import astrodata
from astrodata.testing import download_from_archive
from gempy.utils import logutils
from geminidr.gemini.primitives_gemini import Gemini
from geminidr.f2.primitives_f2 import F2

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

# First obs in sequence, number in sequence, required tolerance
# The tolerance is larger for GNIRS because of the pixel scale uncertainty
# The same is true of F2 actually, there's a 5" difference along the entire
# 6' slit
WCS_DATASETS = [("N20190120S0282", 5, 1.0),  # NIRI+AO PA=0
                ("N20191204S0170", 5, 1.0),  # NIRI (not AO) PA=0
                ("S20160102S0082", 5, 1.0),  # F2 PA=0
                ("N20200119S0150", 4, 2.0),  # GNIRS PA=90
                ("N20200119S0063", 8, 2.0),  # GNIRS PA=110
                ("N20080821S0108", 7, 0.1),  # NIRI+AO PA=27.75
                ("S20170308S0070", 4, 0.75),  # F2 LS (ABBA) PA=135
                ("N20120105S0104", 4, 1.25),  # GNIRS LS (ABBA) PA=15
                ]


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
def niri_sequence(astrofaker):
    """Creates a 3x3 NIRI dither sequence but does not update the WCS;
    only the offsets show the dither"""
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


@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("dataset", WCS_DATASETS[-1:])
def test_standardize_wcs_create_new(dataset):
    """Create a completely new WCS for a dither pattern and confirm that
    the orientation/pixel scales are approximately correct."""
    start = int(dataset[0][10:14])
    filenames = [f"{dataset[0][:10]}{{:04d}}.fits".format(i)
                 for i in range(start, start+dataset[1])]
    files = [download_from_archive(f) for f in filenames]
    adinputs = [astrodata.open(f) for f in files]

    # Remove third dimension
    if adinputs[0].instrument() == "F2":
        p = F2(adinputs)
        p.standardizeStructure()

    if 'IMAGE' in adinputs[0].tags:
        # Create 3x3 grid of pixel locations at corners and centre
        slices = [slice(None, l + 1, l // 2) for l in adinputs[0][0].shape]
        y, x = np.mgrid[slices]
    else:
        # 10 positions along the slit
        x = np.linspace(0, adinputs[0][0].shape[1], 10)
        y = np.full_like(x, adinputs[0][0].shape[0] // 2)

    coords = [ad[0].wcs(x, y) for ad in adinputs]
    coords1 = [[SkyCoord(ra, dec, unit='deg') for ra, dec in zip(*c)] for c in coords]

    # NB. Because this is the Gemini version, the WCS remains in "imaging" form
    p = Gemini(adinputs)
    p.standardizeWCS(bad_wcs="new", debug_max_deadtime=2000)

    new_coords = [ad[0].wcs(x, y) for ad in p.streams['main']]
    coords2 = [[SkyCoord(ra, dec, unit='deg') for ra, dec in zip(*c)] for c in new_coords]

    # Compare WCS coords of those pixels in the pre- and post-modified ADs
    # and check that the standard deviation is small. If the PA or pixel scale
    # has been used incorrectly, the central point will be OK but the edge
    # points will be highly offset. We cannot use the maximum separation since
    # there may be an absolute offset.
    for c1, c2 in zip(coords1, coords2):
        separations = [cc1.separation(cc2).arcsec for cc1, cc2 in zip(c1, c2)]
        assert np.std(separations) < dataset[2]