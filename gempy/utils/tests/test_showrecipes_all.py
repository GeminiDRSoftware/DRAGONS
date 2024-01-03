import pytest

from astrodata.testing import download_from_archive
from gempy.utils.showrecipes import showrecipes

GNIRS_SPECT = "N20190206S0279.fits"
GNIRS = 'N20180101S0028.fits'
GMOS = 'S20180223S0227.fits'
GMOS_NS = 'S20171116S0078.fits'
GMOS_SPECT = "N20110826S0336.fits"
# NIFS = 'N20160727S0077.fits'
# GRACES = 'N20190116G0054i.fits'
NIRI = 'N20190120S0287.fits'
NIRI_LS = 'N20050918S0134.fits'
GSAOI_DARK = 'S20150609S0023.fits'
GSAOI_IMAGE = 'S20170505S0095.fits'
GSAOI_FLAT = 'S20170505S0031.fits'


@pytest.mark.dragons_remote_data
def test_showrecipes_on_gnirs():
    file_location = download_from_archive(GNIRS)

    expected_answers = [
        "Input file: {}".format(file_location),
        "Input tags: ",
        "Recipes available for the input file:",
        "   geminidr.gnirs.recipes.sq.recipes_IMAGE::alignAndStack",
        "   geminidr.gnirs.recipes.sq.recipes_IMAGE::reduce",
        "   geminidr.gnirs.recipes.qa.recipes_IMAGE::reduce",
        "   geminidr.gnirs.recipes.sq.recipes_IMAGE::alignAndStack",
        "   geminidr.gnirs.recipes.sq.recipes_IMAGE::reduce",
    ]

    answer = showrecipes(file_location)

    for i in range(len(expected_answers)):
        assert expected_answers[i] in answer


@pytest.mark.dragons_remote_data
def test_showrecipes_on_gnirs_spect():
    file_location = download_from_archive(GNIRS_SPECT)

    gnirs_spect_answer = [
        "Input file: {}".format(file_location),
        "Input tags: ",
        "!!! No recipes were found for this file !!!"
    ]

    answer = showrecipes(file_location)

    for i in range(len(gnirs_spect_answer)):
        assert gnirs_spect_answer[i] in answer


@pytest.mark.dragons_remote_data
def test_showrecipes_on_gmos():
    file_location = download_from_archive(GMOS)

    gmos_answer = [
        "Input file: {}".format(file_location),
        "Input tags: ",
        "Recipes available for the input file: ",
        "   geminidr.gmos.recipes.sq.recipes_IMAGE::makeProcessedFringe",
        "   geminidr.gmos.recipes.sq.recipes_IMAGE::reduce",
        "   geminidr.gmos.recipes.qa.recipes_IMAGE::makeProcessedFringe",
        "   geminidr.gmos.recipes.qa.recipes_IMAGE::reduce",
        "   geminidr.gmos.recipes.qa.recipes_IMAGE::reduce_nostack",
        "   geminidr.gmos.recipes.qa.recipes_IMAGE::stack"
    ]

    answer = showrecipes(file_location)

    for i in range(len(gmos_answer)):
        assert gmos_answer[i] in answer

@pytest.mark.dragons_remote_data
def test_showrecipes_on_gmos_ns():
    file_location = download_from_archive(GMOS_NS)

    gmos_ns_answer = [
        "Input file: {}".format(file_location),
        "Input tags: ",
        "Recipes available for the input file: ",
        "   geminidr.gmos.recipes.qa.recipes_NS::reduce"
    ]

    answer = showrecipes(file_location)

    for i in range(len(gmos_ns_answer)):
        assert gmos_ns_answer[i] in answer


@pytest.mark.dragons_remote_data
def test_showrecipes_on_gmos_spect():
    file_location = download_from_archive(GMOS_SPECT)

    gmos_spect_answer = [
        "Input file: {}".format(file_location),
        "Input tags: ",
        "!!! No recipes were found for this file !!!",
    ]

    answer = showrecipes(file_location)

    for i in range(len(gmos_spect_answer)):
        assert gmos_spect_answer[i] in answer


@pytest.mark.dragons_remote_data
def test_showrecipes_on_gsaoi_dark():
    file_location = download_from_archive(GSAOI_DARK)

    gsaoi_dark_answer = [
        "Input file: {}".format(file_location),
        "Input tags: ",
        "Recipes available for the input file: ",
        "   geminidr.gsaoi.recipes.sq.recipes_DARK::makeProcessedDark"
    ]

    answer = showrecipes(file_location)

    for i in range(len(gsaoi_dark_answer)):
        assert gsaoi_dark_answer[i] in answer


@pytest.mark.dragons_remote_data
def test_showrecipes_on_gsaoi_image():
    file_location = download_from_archive(GSAOI_IMAGE)

    gsaoi_image_answer = [
        "Input file: {}".format(file_location),
        "Input tags: ",
        "Recipes available for the input file: ",
        "   geminidr.gsaoi.recipes.sq.recipes_IMAGE::reduce_nostack",
        "   geminidr.gsaoi.recipes.qa.recipes_IMAGE::reduce_nostack"]

    answer = showrecipes(file_location)

    for i in range(len(gsaoi_image_answer)):
        assert gsaoi_image_answer[i] in answer


@pytest.mark.dragons_remote_data
def test_showrecipes_on_gsaoi_flat():
    file_location = download_from_archive(GSAOI_FLAT)

    gsaoi_flat_answer = [
        "Input file: {}".format(file_location),
        "Input tags: ",
        "Recipes available for the input file: ",
        "   geminidr.gsaoi.recipes.sq.recipes_FLAT_IMAGE::makeProcessedBPM",
        "   geminidr.gsaoi.recipes.sq.recipes_FLAT_IMAGE::makeProcessedFlat",
        "   geminidr.gsaoi.recipes.qa.recipes_FLAT_IMAGE::makeProcessedFlat"
    ]
    answer = showrecipes(file_location)

    for i in range(len(gsaoi_flat_answer)):
        assert gsaoi_flat_answer[i] in answer


@pytest.mark.dragons_remote_data
def test_showrecipes_on_niri():
    file_location = download_from_archive(NIRI)

    niri_answer = [
        "Input file: {}".format(file_location),
        "Input tags: ",
        "Recipes available for the input file: ",
        "   geminidr.niri.recipes.sq.recipes_IMAGE::makeSkyFlat",
        "   geminidr.niri.recipes.sq.recipes_IMAGE::reduce",
        "   geminidr.niri.recipes.qa.recipes_IMAGE::makeSkyFlat",
        "   geminidr.niri.recipes.qa.recipes_IMAGE::reduce"
    ]

    answer = showrecipes(file_location)

    for i in range(len(niri_answer)):
        assert niri_answer[i] in answer


@pytest.mark.dragons_remote_data
def test_showrecipes_on_f2():
    file_location = download_from_archive(F2)

    f2_answer = [
        "Input file: {}".format(file_location),
        "Input tags: ",
        "Recipes available for the input file: ",
        "   geminidr.f2.recipes.sq.recipes_IMAGE::makeSkyFlat",
        "   geminidr.f2.recipes.sq.recipes_IMAGE::reduce",
        "   geminidr.f2.recipes.qa.recipes_IMAGE::reduce",
        "   geminidr.f2.recipes.qa.recipes_IMAGE::reduce_nostack"
    ]

    answer = showrecipes(file_location)

    for i in range(len(f2_answer)):
        assert f2_answer[i] in answer

#
# # Creates a list of the files and answers, in same order so they can be parsed
# files = [GNIRS_SPECT, GMOS_SPECT, GSAOI_DARK, GSAOI_IMAGE, GSAOI_FLAT,
#          GMOS_NS, GNIRS, GMOS, NIRI, F2, NIFS, GRACES]
#
# answers = [gnirs_spect_answer, gmos_spect_answer, gsaoi_dark_answer,
#            gsaoi_image_answer, gsaoi_flat_answer, gmos_ns_answer,
#            gnirs_answer, gmos_answer, niri_answer, f2_answer,
#            "ImportError", "ImportError"]
#
#
# def test_showrecipes_with_all_instruments(path_to_inputs):
#     for i in range(len(files)):
#         try:
#             for t in range(len(answers[i])):
#                 file_location = path_to_inputs + 'Gempy/' + files[i]
#                 answer = showrecipes(file_location)
#                 assert answers[i] == answer
#         except ImportError:
#             if answers[i] == 'ImportError':
#                 pass
#             else:
#                 raise ImportError
