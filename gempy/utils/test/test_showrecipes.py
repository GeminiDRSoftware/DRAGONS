
import os
import glob
import pytest
import warnings
import astrodata
import gemini_instruments

from gempy.utils.showrecipes import showrecipes


try:
    path = os.environ['TEST_PATH']
except KeyError:
    warnings.warn("Could not find environment variable: $TEST_PATH")
    path = ''

if not os.path.exists(path):
    warnings.warn("Could not find path stored in $TEST_PATH: {}".format(path))
    path = ''


path = path + "Gempy/"


GNIRS = "S20171208S0054.fits"
GNIRS_SPECT = "N20190206S0279.fits"
GMOS = 'S20180223S0229.fits'
GMOS_NS = 'S20170111S0166.fits'
GMOS_SPECT = "N20110826S0336.fits"
NIRI = "N20190120S0287.fits"
F2 = "S20190213S0084.fits"
NIFS = 'N20160727S0077.fits'
GRACES = 'N20190116G0054i.fits'
GSAOI_DARK = 'S20150609S0023.fits'
GSAOI_IMAGE = 'S20170505S0095.fits'
GSAOI_FLAT = 'S20170505S0031.fits'


gnirs_answer = [
    "Input file: {}".format(os.path.normpath(os.path.join(path, GNIRS))),
    "Input tags: ",
    "'FLAT', 'AZEL_TARGET', 'IMAGE', 'DOMEFLAT', 'GSAOI', 'LAMPON', ",
    "'RAW', 'GEMINI', 'NON_SIDEREAL', 'CAL', 'UNPREPARED', 'SOUTH'",
    "Recipes available for the input file: ",
    "   geminidr.gsaoi.recipes.sq.recipes_FLAT_IMAGE::makeProcessedBPM",
    "   geminidr.gsaoi.recipes.sq.recipes_FLAT_IMAGE::makeProcessedFlat",
    "   geminidr.gsaoi.recipes.qa.recipes_FLAT_IMAGE::makeProcessedFlat"]


def test_showrecipes_on_gnirs(test_path):
    file_location = test_path + 'Gempy/' + GNIRS
    answer = showrecipes(file_location)
    for i in range(len(gnirs_answer)):
        assert gnirs_answer[i] in answer


gnirs_spect_answer = [
    "Input file: {}".format(os.path.normpath(
        os.path.join(path, GNIRS_SPECT))),
    "Input tags: ", 
    "'RAW', 'GEMINI', 'NORTH', 'SIDEREAL', 'GNIRS', ",
    "'UNPREPARED', 'SPECT', 'XD'",
    "!!! No recipes were found for this file !!!"]


def test_showrecipes_on_gnirs_spect(test_path):
    file_location = test_path + 'Gempy/' + GNIRS_SPECT
    answer = showrecipes(file_location)
    for i in range(len(gnirs_spect_answer)):
        assert gnirs_spect_answer[i] in answer


gmos_answer = [
    "Input file: {}".format(os.path.normpath(os.path.join(path, GMOS))),
    "Input tags: ", 
    "'SOUTH', 'RAW', 'GMOS', 'GEMINI', 'SIDEREAL', ",
    "'UNPREPARED', 'IMAGE', 'MASK', 'ACQUISITION'",
    "Recipes available for the input file: ",
    "   geminidr.gmos.recipes.sq.recipes_IMAGE::makeProcessedFringe",
    "   geminidr.gmos.recipes.sq.recipes_IMAGE::reduce",
    "   geminidr.gmos.recipes.qa.recipes_IMAGE::makeProcessedFringe",
    "   geminidr.gmos.recipes.qa.recipes_IMAGE::reduce",
    "   geminidr.gmos.recipes.qa.recipes_IMAGE::reduce_nostack",
    "   geminidr.gmos.recipes.qa.recipes_IMAGE::stack"]


def test_showrecipes_on_gmos(test_path):
    file_location = test_path + 'Gempy/' + GMOS
    answer = showrecipes(file_location)
    for i in range(len(gmos_answer)):
        assert gmos_answer[i] in answer


gmos_ns_answer = [
    "Input file: {}".format(os.path.normpath(os.path.join(path, GMOS_NS))),
    "Input tags: ", 
    "'RAW', 'GMOS', 'GEMINI', 'LS', 'UNPREPARED', ",
    "'SPECT', 'NODANDSHUFFLE', 'SOUTH', 'SIDEREAL'",
    "Recipes available for the input file: ",
    "   geminidr.gmos.recipes.qa.recipes_NS::reduce"]


def test_showrecipes_on_gmos_ns(test_path):
    file_location = test_path + 'Gempy/' + GMOS_NS
    answer = showrecipes(file_location)
    for i in range(len(gmos_ns_answer)):
        assert gmos_ns_answer[i] in answer


gmos_spect_answer = [
    "Input file: {}".format(os.path.normpath(
        os.path.join(path, GMOS_SPECT))),
    "Input tags: ",
    "'RAW', 'GMOS', 'GEMINI', 'NORTH', 'SIDEREAL', ",
    "'UNPREPARED', 'SPECT', 'MOS'",
    "!!! No recipes were found for this file !!!"]


def test_showrecipes_on_gmos_spect(test_path):
    file_location = test_path + 'Gempy/' + GMOS_SPECT
    answer = showrecipes(file_location)
    for i in range(len(gmos_spect_answer)):
        assert gmos_spect_answer[i] in answer


gsaoi_dark_answer = [
    "Input file: {}".format(os.path.normpath(
        os.path.join(path, GSAOI_DARK))),
    "Input tags: ", 
    "'DARK', 'RAW', 'AT_ZENITH', 'AZEL_TARGET', 'CAL', ",
    "'UNPREPARED', 'SOUTH', 'GEMINI', 'GSAOI', 'NON_SIDEREAL'",
    "Recipes available for the input file: ",
    "   geminidr.gsaoi.recipes.sq.recipes_DARK::makeProcessedDark"]


def test_showrecipes_on_gsaoi_dark(test_path):
    file_location = test_path + 'Gempy/' + GSAOI_DARK
    answer = showrecipes(file_location)
    for i in range(len(gsaoi_dark_answer)):
        assert gsaoi_dark_answer[i] in answer


gsaoi_image_answer = [
    "Input file: {}".format(os.path.normpath(
        os.path.join(path, GSAOI_IMAGE))),
    "Input tags: ", 
    "'SOUTH', 'RAW', 'GEMINI', 'SIDEREAL', 'UNPREPARED', ",
    "'IMAGE', 'GSAOI'",
    "Recipes available for the input file: ",
    "   geminidr.gsaoi.recipes.sq.recipes_IMAGE::reduce_nostack",
    "   geminidr.gsaoi.recipes.qa.recipes_IMAGE::reduce_nostack"]


def test_showrecipes_on_gsaoi_image(test_path):
    file_location = test_path + 'Gempy/' + GSAOI_IMAGE
    answer = showrecipes(file_location)
    for i in range(len(gsaoi_image_answer)):
        assert gsaoi_image_answer[i] in answer


gsaoi_flat_answer = [
    "Input file: {}".format(os.path.normpath(
        os.path.join(path, GSAOI_FLAT))),
    "Input tags: ", 
    "'FLAT', 'AZEL_TARGET', 'IMAGE', 'DOMEFLAT', ",
    "'GSAOI', 'LAMPON', 'RAW', 'GEMINI', 'NON_SIDEREAL'",
    "'CAL', 'UNPREPARED', 'SOUTH'",
    "Recipes available for the input file: ",
    "   geminidr.gsaoi.recipes.sq.recipes_FLAT_IMAGE::makeProcessedBPM",
    "   geminidr.gsaoi.recipes.sq.recipes_FLAT_IMAGE::makeProcessedFlat",
    "   geminidr.gsaoi.recipes.qa.recipes_FLAT_IMAGE::makeProcessedFlat"]


def test_showrecipes_on_gsaoi_flat(test_path):
    file_location = test_path + 'Gempy/' + GSAOI_FLAT
    answer = showrecipes(file_location)
    for i in range(len(gsaoi_flat_answer)):
        assert gsaoi_flat_answer[i] in answer


niri_answer = [
    "Input file: {}".format(os.path.normpath(os.path.join(path, NIRI))),
    "Input tags: ", 
    "'RAW', 'GEMINI', 'NORTH', 'SIDEREAL', 'UNPREPARED', 'IMAGE', 'NIRI'",
    "Recipes available for the input file: ",
    "   geminidr.niri.recipes.sq.recipes_IMAGE::makeSkyFlat",
    "   geminidr.niri.recipes.sq.recipes_IMAGE::reduce",
    "   geminidr.niri.recipes.qa.recipes_IMAGE::makeSkyFlat",
    "   geminidr.niri.recipes.qa.recipes_IMAGE::reduce"]


def test_showrecipes_on_niri(test_path):
    file_location = test_path + 'Gempy/' + NIRI
    answer = showrecipes(file_location)
    for i in range(len(niri_answer)):
        assert niri_answer[i] in answer


f2_answer = [
    "Input file: {}".format(os.path.normpath(os.path.join(path, F2))),
    "Input tags: ", 
    "'SOUTH', 'RAW', 'F2', 'GEMINI', 'SIDEREAL', ",
    "'UNPREPARED', 'IMAGE', 'ACQUISITION'",
    "Recipes available for the input file: ",
    "   geminidr.f2.recipes.sq.recipes_IMAGE::makeSkyFlat",
    "   geminidr.f2.recipes.sq.recipes_IMAGE::reduce",
    "   geminidr.f2.recipes.qa.recipes_IMAGE::reduce",
    "   geminidr.f2.recipes.qa.recipes_IMAGE::reduce_nostack"]


def test_showrecipes_on_f2(test_path):
    file_location = test_path + 'Gempy/' + F2
    answer = showrecipes(file_location)
    for i in range(len(f2_answer)):
        assert f2_answer[i] in answer
