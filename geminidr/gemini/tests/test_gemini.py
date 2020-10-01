import copy
import pytest

import astrodata
from gempy.utils import logutils
from geminidr.gemini import primitives_gemini as gemini

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


@pytest.fixture(scope='module', autouse=True)
def setup_log(change_working_dir):
    with change_working_dir():
        logutils.config(file_name='test_gemini.log')


# --- Tests ---
def test_standardize_observatory_headers(gemini_image):

    test_gemini = gemini.Gemini([gemini_image])
    processed_af = test_gemini.standardizeObservatoryHeaders()[0]
    expected_timestamp = processed_af.phu['SDZHDRSG']

    assert isinstance(expected_timestamp, str), "phu SDZHDRSG tag not found!"

    numb_of_extensions = processed_af.phu['NSCIEXT']

    assert isinstance(numb_of_extensions, int), "phu NSCIEXT tag not found!"
    assert (numb_of_extensions == 1), "one science extension expected, more/less found"

    new_af = copy.deepcopy(processed_af)
    test_gemini2 = gemini.Gemini([new_af])
    processed_af2 = test_gemini2.standardizeObservatoryHeaders()[0]
    expected_timestamp2 = processed_af2.phu['SDZHDRSG']
    numb_of_extensions2 = processed_af2.phu['NSCIEXT']

    assert (expected_timestamp == expected_timestamp2)
    assert (numb_of_extensions == numb_of_extensions2)
