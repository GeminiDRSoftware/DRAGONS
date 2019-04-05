
import os
import glob
import pytest
import warnings
import astrodata
import gemini_instruments

from gempy.utils.show_recipes import show_recipes


try:
    path = os.environ['TEST_PATH']
except KeyError:
    warnings.warn("Could not find environment variable: $TEST_PATH")
    path = ''

if not os.path.exists(path):
    warnings.warn("Could not find path stored in $TEST_PATH: {}".format(path))
    path = ''


path = path + "geminidr/"


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


gnirs_answer = \
    "Input file: {}\n".format(os.path.normpath(os.path.join(path, GNIRS))) + \
    "Input tags: set(['FLAT', 'AZEL_TARGET', 'IMAGE', 'DOMEFLAT', 'GSAOI', " \
    "'LAMPON', 'RAW', 'GEMINI', 'NON_SIDEREAL', 'CAL', " \
    "'UNPREPARED', 'SOUTH'])\n" \
    "Recipes available for the input file: \n" \
    "   geminidr.gsaoi.recipes.sq.recipes_FLAT_IMAGE::makeProcessedBPM\n" \
    "   geminidr.gsaoi.recipes.sq.recipes_FLAT_IMAGE::makeProcessedFlat\n"\
    "   geminidr.gsaoi.recipes.qa.recipes_FLAT_IMAGE::makeProcessedFlat"


gnirs_spect_answer = \
    "Input file: {}\n".format(os.path.normpath(
        os.path.join(path, GNIRS_SPECT))) + \
    "Input tags: set(['RAW', 'GEMINI', 'NORTH', 'SIDEREAL', 'GNIRS', " \
    "'UNPREPARED', 'SPECT', 'XD'])\n" \
    "!!! No recipes were found for this file !!!"


gmos_answer = \
    "Input file: {}\n".format(os.path.normpath(os.path.join(path, GMOS))) + \
    "Input tags: set(['SOUTH', 'RAW', 'GMOS', 'GEMINI', 'SIDEREAL', " \
    "'UNPREPARED', 'IMAGE', 'MASK', 'ACQUISITION'])\n" \
    "Recipes available for the input file: \n" \
    "   geminidr.gmos.recipes.sq.recipes_IMAGE::makeProcessedFringe\n" \
    "   geminidr.gmos.recipes.sq.recipes_IMAGE::reduce\n" \
    "   geminidr.gmos.recipes.qa.recipes_IMAGE::makeProcessedFringe\n" \
    "   geminidr.gmos.recipes.qa.recipes_IMAGE::reduce\n" \
    "   geminidr.gmos.recipes.qa.recipes_IMAGE::reduce_nostack\n" \
    "   geminidr.gmos.recipes.qa.recipes_IMAGE::stack"

gmos_ns_answer = \
    "Input file: {}\n".format(os.path.normpath(os.path.join(path, GMOS_NS))) + \
    "Input tags: set(['RAW', 'GMOS', 'GEMINI', 'LS', 'UNPREPARED', " \
    "'SPECT', 'NODANDSHUFFLE', 'SOUTH', 'SIDEREAL'])\n" \
    "Recipes available for the input file: \n" \
    "   geminidr.gmos.recipes.qa.recipes_NS::reduce"

gsaoi_dark_answer = \
    "Input file: {}\n".format(os.path.normpath(
        os.path.join(path, GSAOI_DARK))) + \
    "Input tags: set(['DARK', 'RAW', 'AT_ZENITH', 'AZEL_TARGET', 'CAL', " \
    "'UNPREPARED', 'SOUTH', 'GEMINI', 'GSAOI', 'NON_SIDEREAL'])\n" \
    "Recipes available for the input file: \n" \
    "   geminidr.gsaoi.recipes.sq.recipes_DARK::makeProcessedDark"


gsaoi_image_answer = \
    "Input file: {}\n".format(os.path.normpath(
        os.path.join(path, GSAOI_IMAGE))) + \
    "Input tags: set(['SOUTH', 'RAW', 'GEMINI', 'SIDEREAL', 'UNPREPARED', " \
    "'IMAGE', 'GSAOI'])\n" \
    "Recipes available for the input file: \n" \
    "   geminidr.gsaoi.recipes.sq.recipes_IMAGE::reduce_nostack\n" \
    "   geminidr.gsaoi.recipes.qa.recipes_IMAGE::reduce_nostack"


gsaoi_flat_answer = \
    "Input file: {}\n".format(os.path.normpath(
        os.path.join(path, GSAOI_FLAT))) + \
    "Input tags: set(['FLAT', 'AZEL_TARGET', 'IMAGE', 'DOMEFLAT', " \
    "'GSAOI', 'LAMPON', 'RAW', 'GEMINI', 'NON_SIDEREAL', 'CAL', " \
    "'UNPREPARED', 'SOUTH'])\n" \
    "Recipes available for the input file: \n" \
    "   geminidr.gsaoi.recipes.sq.recipes_FLAT_IMAGE::makeProcessedBPM\n" \
    "   geminidr.gsaoi.recipes.sq.recipes_FLAT_IMAGE::makeProcessedFlat\n" \
    "   geminidr.gsaoi.recipes.qa.recipes_FLAT_IMAGE::makeProcessedFlat"


gmos_spect_answer = \
    "Input file: {}\n".format(os.path.normpath(
        os.path.join(path, GMOS_SPECT))) + \
    "Input tags: set(['RAW', 'GMOS', 'GEMINI', 'NORTH', 'SIDEREAL', " \
    "'UNPREPARED', 'SPECT', 'MOS'])\n" \
    "!!! No recipes were found for this file !!!"

niri_answer = \
    "Input file: {}\n".format(os.path.normpath(os.path.join(path, NIRI))) + \
    "Input tags: set(['RAW', 'GEMINI', 'NORTH', 'SIDEREAL', 'UNPREPARED', " \
    "'IMAGE', 'NIRI'])\n" \
    "Recipes available for the input file: \n" \
    "   geminidr.niri.recipes.sq.recipes_IMAGE::makeSkyFlat\n" \
    "   geminidr.niri.recipes.sq.recipes_IMAGE::reduce\n" \
    "   geminidr.niri.recipes.qa.recipes_IMAGE::makeSkyFlat\n" \
    "   geminidr.niri.recipes.qa.recipes_IMAGE::reduce"

f2_answer = \
    "Input file: {}\n".format(os.path.normpath(os.path.join(path, F2))) + \
    "Input tags: set(['SOUTH', 'RAW', 'F2', 'GEMINI', 'SIDEREAL', " \
    "'UNPREPARED', 'IMAGE', 'ACQUISITION'])\n" \
    "Recipes available for the input file: \n" \
    "   geminidr.f2.recipes.sq.recipes_IMAGE::makeSkyFlat\n" \
    "   geminidr.f2.recipes.sq.recipes_IMAGE::reduce\n" \
    "   geminidr.f2.recipes.qa.recipes_IMAGE::reduce\n" \
    "   geminidr.f2.recipes.qa.recipes_IMAGE::reduce_nostack"


# Creates a list of the files and answers, in same order so they can be parsed
files = [GNIRS_SPECT, GMOS_SPECT, GSAOI_DARK, GSAOI_IMAGE, GSAOI_FLAT,
         GMOS_NS, GNIRS, GMOS, NIRI, F2, NIFS, GRACES]

answers = [gnirs_spect_answer, gmos_spect_answer, gsaoi_dark_answer,
           gsaoi_image_answer, gsaoi_flat_answer, gmos_ns_answer,
           gnirs_answer, gmos_answer, niri_answer, f2_answer,
           "ImportError", "ImportError"]


def test_show_recipes_with_all_instruments(test_path):
    for i in range(len(files)):
        try:
            file_location = test_path + 'geminidr/' + files[i]
            answer = show_recipes(file_location)
            assert answers[i] == answer
        except ImportError:
            if answers[i] == 'ImportError':
                pass
            else:
                raise ImportError
