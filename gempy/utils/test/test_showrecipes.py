#!/usr/bin/env python

import os
import glob
import pytest
import warnings

import geminidr
import astrodata
import gemini_instruments
from recipe_system.utils.errors import ModeError
from recipe_system.utils.errors import RecipeNotFound
from gempy.utils.showrecipes import showprims

try:
    path = os.environ['TEST_PATH']
except KeyError:
    warnings.warn("Could not find environment variable: $TEST_PATH")
    path = ''

if not os.path.exists(path):
    warnings.warn("Could not find path stored in $TEST_PATH: {}".format(path))
    path = ''

path = os.path.join(path, "Gempy")
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


gnirs_answer = [
    "Recipe not provided, default recipe (makeProcessedFlat) will be used.",
    "Input file: {}".format(os.path.normpath(os.path.join(path, GNIRS))),
    "Input tags: ",
    "Input mode: sq",
    "Input recipe: makeProcessedFlat",
    "Matched recipe: geminidr.gsaoi.recipes.sq.recipes_FLAT_IMAGE",
    "::makeProcessedFlat",
    "Recipe location: {}".format(os.path.normpath(os.path.join(
        dragons_location, "gsaoi/recipes/sq/recipes_FLAT_IMAGE.py"))),
    "Recipe tags: ",
    "Primitives used: "]


def test_showprims_on_gnirs(test_path):
    file_location = os.path.join(test_path, 'Gempy/', GNIRS)
    answer = showprims(file_location)

    for i in range(len(gnirs_answer)):
        assert gnirs_answer[i] in answer


def test_showprims_on_gnirs_spect(test_path):
    try:
        file_location = os.path.join(test_path, 'Gempy', GNIRS_SPECT)
        answer = showprims(file_location, 'qa')
        assert "RecipeNotFound Error" in answer

    except RecipeNotFound:
        pass


# # # # # #  GMOS  # # # # # #


gmos_answer = [
    "Recipe not provided, default recipe (reduce) will be used.",
    "Input file: {}".format(os.path.normpath(os.path.join(path, GMOS))),
    "Input tags: ",
    "Input mode: sq",
    "Input recipe: reduce",
    "Matched recipe: geminidr.gmos.recipes.sq.recipes_IMAGE::reduce",
    "Recipe location: {}".format(os.path.normpath(os.path.join(
        dragons_location + "gmos/recipes/sq/recipes_IMAGE.py"))),
    "Recipe tags: ",
    "Primitives used: "]


def test_showprims_on_gmos(test_path):
    file_location = os.path.join(test_path, 'Gempy', GMOS)
    answer = showprims(file_location)

    for i in range(len(gmos_answer)):
        assert gmos_answer[i] in answer


def test_showprims_on_gmos_spect(test_path):
    try:
        file_location = os.path.join(test_path, 'Gempy', GMOS_SPECT)
        answer = showprims(file_location)
        assert "RecipeNotFound Error" in answer

    except RecipeNotFound:
        pass


gmos_ns_answer = [
    "Recipe not provided, default recipe (reduce) will be used.",
    "Input file: {}".format(os.path.normpath(os.path.join(path, GMOS_NS))),
    "Input tags: ",
    "Input mode: qa",
    "Input recipe: reduce",
    "Matched recipe: geminidr.gmos.recipes.qa.recipes_NS::reduce",
    "Recipe location: {}".format(os.path.normpath(os.path.join(
        dragons_location + "gmos/recipes/qa/recipes_NS.py"))),
    "Recipe tags: ",
    "Primitives used: "]


def test_showprims_on_gmos_ns(test_path):
    file_location = os.path.join(test_path, 'Gempy', GMOS_NS)
    answer = showprims(file_location, 'qa')

    for i in range(len(gmos_ns_answer)):
        assert gmos_ns_answer[i] in answer


def test_showprims_on_gmos_spect_default_mode(test_path):
    try:
        file_location = os.path.join(test_path, 'Gempy', GMOS_NS)
        answer = showprims(file_location)
        assert "RecipeNotFound Error" in answer
    except RecipeNotFound:
        pass


# # # # # #  GSAOI  # # # # # #


gsaoi_dark_answer = [
    "Recipe not provided, default recipe (makeProcessedDark) will be used.",
    "Input file: {}".format(os.path.normpath(
        os.path.join(path, GSAOI_DARK))),
    "Input tags: ",
    "Input mode: sq",
    "Input recipe: makeProcessedDark",
    "Matched recipe: geminidr.gsaoi.recipes.sq.recipes_DARK",
    "::makeProcessedDark",
    "Recipe location: {}".format(os.path.normpath(os.path.join(
        dragons_location + "gsaoi/recipes/sq/recipes_DARK.py"))),
    "Recipe tags: ",
    "Primitives used: "]


def test_showprims_on_gsaoi_dark(test_path):
    file_location = os.path.join(test_path, 'Gempy', GSAOI_DARK)
    answer = showprims(file_location, 'sq', 'default')

    for i in range(len(gsaoi_dark_answer)):
        assert gsaoi_dark_answer[i] in answer


def test_showprims_on_gsaoi_dark_qa_mode(test_path):
    try:
        file_location = os.path.join(test_path, 'Gempy', GSAOI_DARK)
        answer = showprims(file_location, 'qa')
        assert "RecipeNotFound Error" in answer
    except RecipeNotFound:
        pass


gsaoi_image_answer_sq = [
    "Recipe not provided, default recipe (reduce_nostack) will be used.",
    "Input file: {}".format(os.path.normpath(
        os.path.join(path, GSAOI_IMAGE))),
    "Input tags: ",
    "Input mode: sq",
    "Input recipe: reduce_nostack",
    "Matched recipe: geminidr.gsaoi.recipes.sq.recipes_IMAGE::reduce_nostack",
    "Recipe location: {}".format(os.path.normpath(os.path.join(
        dragons_location + "gsaoi/recipes/sq/recipes_IMAGE.py"))),
    "Recipe tags: ",
    "Primitives used: "]


def test_showprims_on_gsaoi_image_sq_mode(test_path):
    file_location = os.path.join(test_path, 'Gempy', GSAOI_IMAGE)
    answer = showprims(file_location)
    for i in range(len(gsaoi_image_answer_sq)):
        assert gsaoi_image_answer_sq[i] in answer


gsaoi_image_answer_qa = [
    "Input file: {}".format(os.path.normpath(
        os.path.join(path, GSAOI_IMAGE))),
    "Input tags: ",
    "Input mode: qa",
    "Input recipe: reduce_nostack",
    "Matched recipe: geminidr.gsaoi.recipes.qa.recipes_IMAGE::reduce_nostack",
    "Recipe location: {}".format(os.path.normpath(os.path.join(
        dragons_location + "gsaoi/recipes/qa/recipes_IMAGE.py"))),
    "Recipe tags: ",
    "Primitives used: "]


def test_showprims_on_gsaoi_image_qa_mode(test_path):
    file_location = os.path.join(test_path, 'Gempy', GSAOI_IMAGE)
    answer = showprims(file_location, 'qa', 'reduce_nostack')
    for i in range(len(gsaoi_image_answer_qa)):
        assert gsaoi_image_answer_qa[i] in answer


gsaoi_flat_answer = [
    "Recipe not provided, default recipe (makeProcessedFlat) will be used.",
    "Input file: {}".format(os.path.normpath(
        os.path.join(path, GSAOI_FLAT))),
    "Input tags: ",
    "Input mode: sq",
    "Input recipe: makeProcessedFlat",
    "Matched recipe: geminidr.gsaoi.recipes.sq.recipes_FLAT_IMAGE::",
    "makeProcessedFlat",
    "Recipe location: {}".format(os.path.normpath(os.path.join(
        dragons_location + "gsaoi/recipes/sq/recipes_FLAT_IMAGE.py"))),
    "Recipe tags: ",
    "Primitives used: "]


def test_showprims_on_gsaoi_flat(test_path):
    file_location = os.path.join(test_path, 'Gempy', GSAOI_FLAT)
    answer = showprims(file_location)
    for i in range(len(gsaoi_flat_answer)):
        assert gsaoi_flat_answer[i] in answer


def test_showprims_on_gsaoi_flat_ql_mode(test_path):
    try:
        file_location = os.path.join(test_path, 'Gempy', GSAOI_FLAT)
        answer = showprims(file_location, 'ql')
        assert "ModuleNotFoundError" in answer
    except ModeError:
        pass


# # # # # #  NIRI  # # # # # #


niri_answer = [
    "Recipe not provided, default recipe (reduce) will be used.",
    "Input file: {}".format(os.path.normpath(os.path.join(path, NIRI))),
    "Input tags: ",
    "Input mode: sq",
    "Input recipe: reduce",
    "Matched recipe: geminidr.niri.recipes.sq.recipes_IMAGE::reduce",
    "Recipe location: {}".format(os.path.normpath(os.path.join(
        dragons_location + "niri/recipes/sq/recipes_IMAGE.py"))),
    "Recipe tags: ",
    "Primitives used:"]


def test_showprims_on_niri(test_path):
    file_location = os.path.join(test_path, 'Gempy', NIRI)
    answer = showprims(file_location)
    for i in range(len(niri_answer)):
        assert niri_answer[i] in answer


# # # # # #  F2  # # # # # #


f2_answer = [
    "Recipe not provided, default recipe (reduce) will be used.",
    "Input file: {}".format(os.path.normpath(os.path.join(path, F2))),
    "Input tags: ",
    "Input mode: sq",
    "Input recipe: reduce",
    "Matched recipe: geminidr.f2.recipes.sq.recipes_IMAGE::reduce",
    "Recipe location: {}".format(os.path.normpath(os.path.join(
        dragons_location + "f2/recipes/sq/recipes_IMAGE.py"))),
    "Recipe tags: ",
    "Primitives used: "]


def test_showprims_on_f2(test_path):
    file_location = os.path.join(test_path, 'Gempy', F2)
    answer = showprims(file_location)
    for i in range(len(f2_answer)):
        assert f2_answer[i] in answer
