import os
import glob
import pytest
import warnings

import geminidr
import astrodata
import gemini_instruments
from recipe_system.utils.errors import ModeError
from recipe_system.utils.errors import RecipeNotFound
from gempy.utils.show_primitives import show_primitives



try:
    path = os.environ['TEST_PATH']
except KeyError:
    warnings.warn("Could not find environment variable: $TEST_PATH")
    path = ''

if not os.path.exists(path):
    warnings.warn("Could not find path stored in $TEST_PATH: {}".format(path))
    path = ''


path = path + "geminidr/"
dragons_location = '/'.join(geminidr.__file__.split("/")[:-1]) + '/'

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


# # # # # #  GNIRS  # # # # # #


gnirs_answer = \
    "Recipe not provided, default recipe (makeProcessedFlat) will be used.\n" \
    "Input file: {}\n".format(os.path.normpath(os.path.join(path, GNIRS))) + \
    "Input tags: set(['FLAT', 'AZEL_TARGET', 'IMAGE', 'DOMEFLAT', " \
    "'GSAOI', 'LAMPON', 'RAW', 'GEMINI', 'NON_SIDEREAL', " \
    "'CAL', 'UNPREPARED', 'SOUTH'])\n" \
    "Input mode: sq\n" \
    "Input recipe: makeProcessedFlat\n" \
    "Matched recipe: geminidr.gsaoi.recipes.sq.recipes_FLAT_IMAGE" \
    "::makeProcessedFlat\n" \
    "Recipe location: {}\n".format(os.path.normpath(os.path.join(
            dragons_location + "gsaoi/recipes/sq/recipes_FLAT_IMAGE.py"))) + \
    "Recipe tags: set(['FLAT', 'IMAGE', 'GSAOI', 'CAL'])\n" \
    "Primitives used: \n" \
    "   p.prepare()\n" \
    "   p.addDQ()\n" \
    "   p.nonlinearityCorrect()\n" \
    "   p.ADUToElectrons()\n" \
    "   p.addVAR(read_noise=True, poisson_noise=True)\n" \
    "   p.makeLampFlat()\n" \
    "   p.normalizeFlat()\n" \
    "   p.thresholdFlatfield()\n" \
    "   p.storeProcessedFlat()" \



def test_show_primitives_on_gnirs(test_path):
    file_location = test_path + 'geminidr/' + GNIRS
    answer = show_primitives(file_location)
    assert gnirs_answer == answer


def test_show_primitives_on_gnirs_spect(test_path):
    try:
        file_location = test_path + 'geminidr/' + GNIRS_SPECT
        answer = show_primitives(file_location, 'qa')
        assert "RecipeNotFound Error" == answer
    except RecipeNotFound:
        pass


# # # # # #  GMOS  # # # # # #


gmos_answer = \
    "Recipe not provided, default recipe (reduce) will be used.\n" \
    "Input file: {}\n".format(os.path.normpath(os.path.join(path, GMOS))) + \
    "Input tags: set(['SOUTH', 'RAW', 'GMOS', 'GEMINI', 'SIDEREAL', " \
    "'UNPREPARED', 'IMAGE', 'MASK', 'ACQUISITION'])\n" \
    "Input mode: sq\n" \
    "Input recipe: reduce\n" \
    "Matched recipe: geminidr.gmos.recipes.sq.recipes_IMAGE::reduce\n" \
    "Recipe location: {}\n".format(os.path.normpath(os.path.join(
        dragons_location + "gmos/recipes/sq/recipes_IMAGE.py"))) + \
    "Recipe tags: set(['GMOS', 'IMAGE'])\n" \
    "Primitives used: \n" \
    "   p.prepare()\n" \
    "   p.addDQ()\n" \
    "   p.addVAR(read_noise=True)\n" \
    "   p.overscanCorrect()\n" \
    "   p.biasCorrect()\n" \
    "   p.ADUToElectrons()\n" \
    "   p.addVAR(poisson_noise=True)\n" \
    "   p.flatCorrect()\n" \
    "   p.makeFringe()\n" \
    "   p.fringeCorrect()\n" \
    "   p.mosaicDetectors()\n" \
    "   p.adjustWCSToReference()\n" \
    "   p.resampleToCommonFrame()\n" \
    "   p.stackFrames()\n" \
    "   p.writeOutputs()"


def test_show_primitives_on_gmos(test_path):
    file_location = test_path + 'geminidr/' + GMOS
    answer = show_primitives(file_location)
    assert gmos_answer == answer


def test_show_primitives_on_gmos_spect(test_path):
    try:
        file_location = test_path + 'geminidr/' + GMOS_SPECT
        answer = show_primitives(file_location)
        assert "RecipeNotFound Error" == answer
    except RecipeNotFound:
        pass


gmos_ns_answer = \
    "Recipe not provided, default recipe (reduce) will be used.\n" \
    "Input file: {}\n".format(os.path.normpath(os.path.join(path, GMOS_NS))) + \
    "Input tags: set(['RAW', 'GMOS', 'GEMINI', 'LS', 'UNPREPARED', " \
    "'SPECT', 'NODANDSHUFFLE', 'SOUTH', 'SIDEREAL'])\n" \
    "Input mode: qa\n" \
    "Input recipe: reduce\n" \
    "Matched recipe: geminidr.gmos.recipes.qa.recipes_NS::reduce\n" \
    "Recipe location: {}\n".format(os.path.normpath(os.path.join(
        dragons_location + "gmos/recipes/qa/recipes_NS.py"))) + \
    "Recipe tags: set(['GMOS', 'NODANDSHUFFLE', 'SPECT'])\n" \
    "Primitives used: \n" \
    "   p.prepare()\n" \
    "   p.addDQ()\n" \
    "   p.addVAR(read_noise=True)\n" \
    "   p.overscanCorrect()\n" \
    "   p.biasCorrect()\n" \
    "   p.ADUToElectrons()\n" \
    "   p.addVAR(poisson_noise=True)\n" \
    "   p.findAcquisitionSlits()\n" \
    "   p.skyCorrectNodAndShuffle()\n" \
    "   p.measureIQ(display=True)\n" \
    "   p.writeOutputs()"


def test_show_primitives_on_gmos_ns(test_path):
    file_location = test_path + 'geminidr/' + GMOS_NS
    answer = show_primitives(file_location, 'qa')
    assert gmos_ns_answer == answer


def test_show_primitives_on_gmos_spect_default_mode(test_path):
    try:
        file_location = test_path + 'geminidr/' + GMOS_NS
        answer = show_primitives(file_location)
        assert "RecipeNotFound Error" == answer
    except RecipeNotFound:
        pass


# # # # # #  GSAOI  # # # # # #


gsaoi_dark_answer = \
    "Recipe not provided, default recipe (makeProcessedDark) will be used.\n" \
    "Input file: {}\n".format(os.path.normpath(
        os.path.join(path, GSAOI_DARK))) + \
    "Input tags: set(['DARK', 'RAW', 'AT_ZENITH', 'AZEL_TARGET', 'CAL', " \
    "'UNPREPARED', 'SOUTH', 'GEMINI', 'GSAOI', 'NON_SIDEREAL'])\n" \
    "Input mode: sq\n" \
    "Input recipe: makeProcessedDark\n" \
    "Matched recipe: geminidr.gsaoi.recipes.sq.recipes_DARK" \
    "::makeProcessedDark\n" \
    "Recipe location: {}\n".format(os.path.normpath(os.path.join(
        dragons_location + "gsaoi/recipes/sq/recipes_DARK.py"))) + \
    "Recipe tags: set(['DARK', 'GSAOI', 'CAL'])\n" \
    "Primitives used: \n" \
    "   p.prepare()\n" \
    "   p.addDQ(add_illum_mask=False)\n" \
    "   p.addVAR(read_noise=True)\n" \
    "   p.nonlinearityCorrect()\n" \
    "   p.ADUToElectrons()\n" \
    "   p.addVAR(poisson_noise=True)\n" \
    "   p.stackDarks()\n" \
    "   p.storeProcessedDark()"


def test_show_primitives_on_gsaoi_dark(test_path):
    file_location = test_path + 'geminidr/' + GSAOI_DARK
    answer = show_primitives(file_location, 'sq', 'default')
    assert gsaoi_dark_answer == answer


def test_show_primitives_on_gsaoi_dark_qa_mode(test_path):
    try:
        file_location = test_path + 'geminidr/' + GSAOI_DARK
        answer = show_primitives(file_location, 'qa')
        assert "RecipeNotFound Error" == answer
    except RecipeNotFound:
        pass


gsaoi_image_answer_sq = \
    "Recipe not provided, default recipe (reduce_nostack) will be used.\n" \
    "Input file: {}\n".format(os.path.normpath(
        os.path.join(path, GSAOI_IMAGE))) + \
    "Input tags: set(['SOUTH', 'RAW', 'GEMINI', 'SIDEREAL', " \
    "'UNPREPARED', 'IMAGE', 'GSAOI'])\n" \
    "Input mode: sq\n" \
    "Input recipe: reduce_nostack\n" \
    "Matched recipe: geminidr.gsaoi.recipes.sq.recipes_IMAGE::reduce_nostack\n"\
    "Recipe location: {}\n".format(os.path.normpath(os.path.join(
        dragons_location + "gsaoi/recipes/sq/recipes_IMAGE.py"))) + \
    "Recipe tags: set(['IMAGE', 'GSAOI'])\n" \
    "Primitives used: \n" \
    "   p.prepare()\n" \
    "   p.addDQ()\n" \
    "   p.nonlinearityCorrect()\n" \
    "   p.ADUToElectrons()\n" \
    "   p.addVAR(read_noise=True, poisson_noise=True)\n" \
    "   p.flatCorrect()\n" \
    "   p.flushPixels()\n" \
    "   p.separateSky()\n" \
    "   p.associateSky(stream='sky')\n" \
    "   p.skyCorrect(instream='sky', mask_objects=False, outstream='skysub')\n"\
    "   p.detectSources(stream='skysub')\n" \
    "   p.transferAttribute(stream='sky', source='skysub', " \
    "attribute='OBJMASK')\n" \
    "   p.clearStream(stream='skysub')\n" \
    "   p.associateSky()\n" \
    "   p.skyCorrect(mask_objects=True)\n" \
    "   p.writeOutputs()"


def test_show_primitives_on_gsaoi_image_sq_mode(test_path):
    file_location = test_path + 'geminidr/' + GSAOI_IMAGE
    answer = show_primitives(file_location)
    assert gsaoi_image_answer_sq == answer


gsaoi_image_answer_qa = \
    "Input file: {}\n".format(os.path.normpath(
        os.path.join(path, GSAOI_IMAGE))) + \
    "Input tags: set(['SOUTH', 'RAW', 'GEMINI', 'SIDEREAL', " \
    "'UNPREPARED', 'IMAGE', 'GSAOI'])\n" \
    "Input mode: qa\n" \
    "Input recipe: reduce_nostack\n" \
    "Matched recipe: geminidr.gsaoi.recipes.qa.recipes_IMAGE::reduce_nostack\n"\
    "Recipe location: {}\n".format(os.path.normpath(os.path.join(
        dragons_location + "gsaoi/recipes/qa/recipes_IMAGE.py"))) + \
    "Recipe tags: set(['IMAGE', 'GSAOI'])\n" \
    "Primitives used: \n" \
    "   p.prepare()\n" \
    "   p.addDQ()\n" \
    "   p.nonlinearityCorrect()\n" \
    "   p.ADUToElectrons()\n" \
    "   p.addVAR(read_noise=True, poisson_noise=True)\n" \
    "   p.flatCorrect()\n" \
    "   p.detectSources(detect_thresh=5., analysis_thresh=5., back_size=128)\n"\
    "   p.measureIQ(display=True)\n" \
    "   p.writeOutputs()\n" \
    "   p.measureBG()\n" \
    "   p.addReferenceCatalog()\n" \
    "   p.determineAstrometricSolution()\n" \
    "   p.measureCC()\n" \
    "   p.addToList(purpose='forSky')\n" \
    "   p.getList(purpose='forSky', max_frames=9)\n" \
    "   p.separateSky()\n" \
    "   p.associateSky()\n" \
    "   p.skyCorrect()\n" \
    "   p.detectSources(detect_thresh=5., analysis_thresh=5., back_size=128)\n"\
    "   p.measureIQ(display=True)\n" \
    "   p.determineAstrometricSolution()\n" \
    "   p.measureCC()\n" \
    "   p.writeOutputs()"


def test_show_primitives_on_gsaoi_image_qa_mode(test_path):
    file_location = test_path + 'geminidr/' + GSAOI_IMAGE
    answer = show_primitives(file_location, 'qa', 'reduce_nostack')
    assert gsaoi_image_answer_qa == answer


gsaoi_flat_answer = \
    "Recipe not provided, default recipe (makeProcessedFlat) will be used.\n" \
    "Input file: {}\n".format(os.path.normpath(
        os.path.join(path, GSAOI_FLAT))) + \
    "Input tags: set(['FLAT', 'AZEL_TARGET', 'IMAGE', " \
    "'DOMEFLAT', 'GSAOI', 'LAMPON', 'RAW', 'GEMINI', 'NON_SIDEREAL', " \
    "'CAL', 'UNPREPARED', 'SOUTH'])\n" \
    "Input mode: sq\n" \
    "Input recipe: makeProcessedFlat\n" \
    "Matched recipe: geminidr.gsaoi.recipes.sq.recipes_FLAT_IMAGE::" \
    "makeProcessedFlat\n" \
    "Recipe location: {}\n".format(os.path.normpath(os.path.join(
        dragons_location + "gsaoi/recipes/sq/recipes_FLAT_IMAGE.py"))) + \
    "Recipe tags: set(['FLAT', 'IMAGE', 'GSAOI', 'CAL'])\n" \
    "Primitives used: \n" \
    "   p.prepare()\n" \
    "   p.addDQ()\n" \
    "   p.nonlinearityCorrect()\n" \
    "   p.ADUToElectrons()\n" \
    "   p.addVAR(read_noise=True, poisson_noise=True)\n" \
    "   p.makeLampFlat()\n" \
    "   p.normalizeFlat()\n" \
    "   p.thresholdFlatfield()\n" \
    "   p.storeProcessedFlat()"


def test_show_primitives_on_gsaoi_flat(test_path):
    file_location = test_path + 'geminidr/' + GSAOI_FLAT
    answer = show_primitives(file_location)
    assert gsaoi_flat_answer == answer


def test_show_primitives_on_gsaoi_flat_ql_mode(test_path):
    try:
        file_location = test_path + 'geminidr/' + GSAOI_FLAT
        answer = show_primitives(file_location, 'ql')
        assert "ModuleNotFoundError" == answer
    except ModeError:
        pass


# # # # # #  NIRI  # # # # # #


niri_answer = \
    "Recipe not provided, default recipe (reduce) will be used.\n" \
    "Input file: {}\n".format(os.path.normpath(os.path.join(path, NIRI))) + \
    "Input tags: set(['RAW', 'GEMINI', 'NORTH', 'SIDEREAL', " \
    "'UNPREPARED', 'IMAGE', 'NIRI'])\n" \
    "Input mode: sq\n" \
    "Input recipe: reduce\n" \
    "Matched recipe: geminidr.niri.recipes.sq.recipes_IMAGE::reduce\n" \
    "Recipe location: {}\n".format(os.path.normpath(os.path.join(
        dragons_location + "niri/recipes/sq/recipes_IMAGE.py"))) + \
    "Recipe tags: set(['IMAGE', 'NIRI'])\n" \
    "Primitives used: \n" \
    "   p.prepare()\n" \
    "   p.addDQ()\n" \
    "   p.removeFirstFrame()\n" \
    "   p.ADUToElectrons()\n" \
    "   p.addVAR(read_noise=True, poisson_noise=True)\n" \
    "   p.nonlinearityCorrect()\n" \
    "   p.darkCorrect()\n" \
    "   p.flatCorrect()\n" \
    "   p.separateSky()\n" \
    "   p.associateSky(stream='sky')\n" \
    "   p.skyCorrect(instream='sky', mask_objects=False, outstream='skysub')\n"\
    "   p.detectSources(stream='skysub')\n" \
    "   p.transferAttribute(stream='sky', source='skysub', " \
    "attribute='OBJMASK')\n" \
    "   p.clearStream(stream='skysub')\n" \
    "   p.associateSky()\n" \
    "   p.skyCorrect(mask_objects=True)\n" \
    "   p.detectSources()\n" \
    "   p.adjustWCSToReference()\n" \
    "   p.resampleToCommonFrame()\n" \
    "   p.stackFrames()\n" \
    "   p.writeOutputs()"


def test_show_primitives_on_niri(test_path):
    file_location = test_path + 'geminidr/' + NIRI
    answer = show_primitives(file_location)
    assert niri_answer == answer


# # # # # #  F2  # # # # # #


f2_answer = \
    "Recipe not provided, default recipe (reduce) will be used.\n" \
    "Input file: {}\n".format(os.path.normpath(os.path.join(path, F2))) + \
    "Input tags: set(['SOUTH', 'RAW', 'F2', 'GEMINI', 'SIDEREAL', " \
    "'UNPREPARED', 'IMAGE', 'ACQUISITION'])\n" \
    "Input mode: sq\n" \
    "Input recipe: reduce\n" \
    "Matched recipe: geminidr.f2.recipes.sq.recipes_IMAGE::reduce\n" \
    "Recipe location: {}\n".format(os.path.normpath(os.path.join(
        dragons_location + "f2/recipes/sq/recipes_IMAGE.py"))) + \
    "Recipe tags: set(['F2', 'IMAGE'])\n" \
    "Primitives used: \n" \
    "   p.prepare()\n" \
    "   p.addDQ()\n" \
    "   p.ADUToElectrons()\n" \
    "   p.addVAR(read_noise=True, poisson_noise=True)\n" \
    "   p.nonlinearityCorrect()\n" \
    "   p.darkCorrect()\n" \
    "   p.flatCorrect()\n" \
    "   p.flushPixels()\n" \
    "   p.separateSky()\n" \
    "   p.associateSky(stream='sky')\n" \
    "   p.skyCorrect(instream='sky', mask_objects=False, outstream='skysub')\n"\
    "   p.detectSources(stream='skysub')\n" \
    "   p.transferAttribute(stream='sky', source='skysub', " \
    "attribute='OBJMASK')\n" \
    "   p.clearStream(stream='skysub')\n" \
    "   p.associateSky()\n" \
    "   p.skyCorrect(mask_objects=True)\n" \
    "   p.flushPixels()\n" \
    "   p.detectSources()\n" \
    "   p.adjustWCSToReference()\n" \
    "   p.resampleToCommonFrame()\n" \
    "   p.stackFrames()\n" \
    "   p.writeOutputs()"


def test_show_primitives_on_f2(test_path):
    file_location = test_path + 'geminidr/' + F2
    answer = show_primitives(file_location)
    assert f2_answer == answer
