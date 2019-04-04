import sys
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


# Returns list of all files in the TEST_PATH directory
archive_files = glob.glob(os.path.join(path, "*fits"))

# Separates the directory from the list, helps cleanup code
fits_files = [_file.split('/')[-1] for _file in archive_files]


GNIRS = "S20171208S0054.fits"
GNIRS_SPECT = "N20190206S0279.fits"
GMOS = 'S20180223S0229.fits'
GMOS_SPECT = "N20110826S0336.fits"
NIRI = "N20190120S0287.fits"
F2 = "S20190213S0084.fits"
NIFS = 'N20160727S0077.fits'
GRACES = 'N20190116G0054i.fits'

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



gmos_spect_answer = \
    "Input file: {}\n".format(os.path.normpath(os.path.join(path, GMOS_SPECT))) + \
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


list_of_answers = [gnirs_answer, gnirs_spect_answer, gmos_spect_answer
                   ,niri_answer]


@pytest.mark.parametrize('filename, str_answer', [
    (GNIRS_SPECT, gnirs_spect_answer),
    (GMOS_SPECT, gmos_spect_answer),
    (GNIRS, gnirs_answer),
    (GMOS, gmos_answer),
    (NIRI, niri_answer),
    (F2, f2_answer),
    (NIFS, "ImportError"),
    (GRACES, "ImportError")
    ])
def test_show_recipes(test_path, filename, str_answer):
    try:
        file_location = test_path + filename
        answer = show_recipes(file_location)
        assert str_answer == answer
    except ImportError:
        if str_answer == 'ImportError':
            pass
        else:
            raise ImportError


