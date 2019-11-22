#!/usr/bin/env python

import pytest
import re

from gempy.utils.showrecipes import showprims
from recipe_system.utils.errors import RecipeNotFound
from astrodata.testing import download_from_archive

GNIRS = "S20171208S0054.fits"
GNIRS_SPECT = "N20190206S0279.fits"
GMOS = 'S20180223S0229.fits'
GMOS_NS = 'S20171116S0078.fits'
GMOS_SPECT = "N20170529S0168.fits"
NIRI = "N20190120S0287.fits"
F2 = "S20190213S0084.fits"
# NIFS = 'N20160727S0077.fits'
# GRACES = 'N20190116G0054i.fits'
GSAOI_DARK = 'S20150609S0023.fits'
GSAOI_IMAGE = 'S20170505S0095.fits'
GSAOI_FLAT = 'S20170505S0031.fits'

TESTS = {
    'GNIRS': (GNIRS, 'sq', '_default', [
        r"Recipe not provided, default recipe \(makeProcessedFlat\) will be used",
        r"Input file: .*/{}".format(GNIRS),
        r"Input tags: ",
        r"Input mode: sq",
        r"Input recipe: makeProcessedFlat",
        r"Matched recipe: geminidr.gsaoi.recipes.sq.recipes_FLAT_IMAGE",
        r"::makeProcessedFlat",
        r"Recipe location: .*/gsaoi/recipes/sq/recipes_FLAT_IMAGE.py",
        r"Recipe tags: ",
        r"Primitives used: "]),
    'GMOS': (GMOS, 'sq', '_default', [
        r"Recipe not provided, default recipe \(reduce\) will be used",
        r"Input file: .*/{}".format(GMOS),
        r"Input tags: ",
        r"Input mode: sq",
        r"Input recipe: reduce",
        r"Matched recipe: geminidr.gmos.recipes.sq.recipes_IMAGE::reduce",
        r"Recipe location: .*/gmos/recipes/sq/recipes_IMAGE.py",
        r"Recipe tags: ",
        r"Primitives used: "]),
    'GMOS_NS': (GMOS_NS, 'qa', '_default', [
        r"Recipe not provided, default recipe \(reduce\) will be used",
        r"Input file: .*/{}".format(GMOS_NS),
        r"Input tags: ",
        r"Input mode: qa",
        r"Input recipe: reduce",
        r"Matched recipe: geminidr.gmos.recipes.qa.recipes_NS::reduce",
        r"Recipe location: .*/gmos/recipes/qa/recipes_NS.py",
        r"Recipe tags: ",
        r"Primitives used: "]),
    'GSAOI_DARK': (GSAOI_DARK, 'sq', '_default', [
        r"Recipe not provided, default recipe \(makeProcessedDark\) will be used",
        r"Input file: .*/{}".format(GSAOI_DARK),
        r"Input tags: ",
        r"Input mode: sq",
        r"Input recipe: makeProcessedDark",
        r"Matched recipe: geminidr.gsaoi.recipes.sq.recipes_DARK",
        r"::makeProcessedDark",
        r"Recipe location: .*/gsaoi/recipes/sq/recipes_DARK.py",
        r"Recipe tags: ",
        r"Primitives used: "]),
    'GSAOI_IMAGE': (GSAOI_IMAGE, 'sq', '_default', [
        r"Recipe not provided, default recipe \(reduce_nostack\) will be used",
        r"Input file: .*/{}".format(GSAOI_IMAGE),
        r"Input tags: ",
        r"Input mode: sq",
        r"Input recipe: reduce_nostack",
        r"Matched recipe: geminidr.gsaoi.recipes.sq.recipes_IMAGE::reduce_nostack",
        r"Recipe location: .*/gsaoi/recipes/sq/recipes_IMAGE.py",
        r"Recipe tags: ",
        r"Primitives used: "]),
    'GSAOI_IMAGE2': (GSAOI_IMAGE, 'sq', 'reduce_nostack', [
        r"Input file: .*/{}".format(GSAOI_IMAGE),
        r"Input tags: ",
        r"Input mode: sq",
        r"Input recipe: reduce_nostack",
        r"Matched recipe: geminidr.gsaoi.recipes.sq.recipes_IMAGE::reduce_nostack",
        r"Recipe location: .*/gsaoi/recipes/sq/recipes_IMAGE.py",
        r"Recipe tags: ",
        r"Primitives used: "]),
    'GSAOI_FLAT': (GSAOI_FLAT, 'sq', '_default', [
        r"Recipe not provided, default recipe \(makeProcessedFlat\) will be used",
        r"Input file: .*/{}".format(GSAOI_FLAT),
        r"Input tags: ",
        r"Input mode: sq",
        r"Input recipe: makeProcessedFlat",
        r"Matched recipe: geminidr.gsaoi.recipes.sq.recipes_FLAT_IMAGE::",
        r"makeProcessedFlat",
        r"Recipe location: .*/gsaoi/recipes/sq/recipes_FLAT_IMAGE.py",
        r"Recipe tags: ",
        r"Primitives used: "]),
    'NIRI': (NIRI, 'sq', '_default', [
        r"Recipe not provided, default recipe \(reduce\) will be used",
        r"Input file: .*/{}".format(NIRI),
        r"Input tags: ",
        r"Input mode: sq",
        r"Input recipe: reduce",
        r"Matched recipe: geminidr.niri.recipes.sq.recipes_IMAGE::reduce",
        r"Recipe location: .*/niri/recipes/sq/recipes_IMAGE.py",
        r"Recipe tags: ",
        r"Primitives used:"]),
    'F2': (F2, 'sq', '_default', [
        r"Recipe not provided, default recipe \(reduce\) will be used.",
        r"Input file: .*/{}".format(F2),
        r"Input tags: ",
        r"Input mode: sq",
        r"Input recipe: reduce",
        r"Matched recipe: geminidr.f2.recipes.sq.recipes_IMAGE::reduce",
        r"Recipe location: .*/f2/recipes/sq/recipes_IMAGE.py",
        r"Recipe tags: ",
        r"Primitives used: "]),
}


@pytest.mark.remote_data
@pytest.mark.parametrize("name", TESTS)
def test_showprims(name):
    filename, mode, recipe, expected = TESTS[name]
    file_location = download_from_archive(filename)
    answer = showprims(file_location, mode=mode, recipe=recipe)
    for line in expected:
        assert re.search(line, answer)


# # # # # #  GNIRS  # # # # # #
@pytest.mark.remote_data
def test_showprims_on_gnirs_spect():
    try:
        file_location = download_from_archive(GNIRS_SPECT)
        answer = showprims(file_location, 'qa')
        assert "RecipeNotFound Error" in answer
    except RecipeNotFound:
        pass


# # # # # #  GMOS  # # # # # #
@pytest.mark.remote_data
def test_showprims_on_gmos_spect():
    file_location = download_from_archive(GMOS_SPECT)
    answer = showprims(file_location)
    assert "geminidr.gmos.recipes.ql.recipes_LS_SPECT::reduce" in answer


# # # # # #  GSAOI  # # # # # #
@pytest.mark.remote_data
def test_showprims_on_gsaoi_dark_qa_mode():
    try:
        file_location = download_from_archive(GSAOI_DARK)
        answer = showprims(file_location, 'qa')
        assert "RecipeNotFound Error" in answer
    except RecipeNotFound:
        pass
