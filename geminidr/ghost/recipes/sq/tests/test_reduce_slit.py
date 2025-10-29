# Tests for the reduction of all types of slitviewer images

import os
import pytest

import astrodata, gemini_instruments
from geminidr.ghost.primitives_ghost_slit import GHOSTSlit
from geminidr.ghost.recipes.sq.recipes_BIAS_SLITV import makeProcessedSlitBias
from geminidr.ghost.recipes.sq.recipes_FLAT_SLITV import makeProcessedSlitFlat
from geminidr.ghost.recipes.sq.recipes_ARC_SLITV import makeProcessedSlitArc
from geminidr.ghost.recipes.sq.recipes_SLITV import makeProcessedSlit
from astrodata.testing import ad_compare


# Input files (and slitbias calibrations if required)
# Allow for multiple datasets for each test
bias_datasets = ["S20230513S0012_slit.fits"]
flat_datasets = [("S20230513S0010_slit.fits", "S20230513S0012_slit_bias.fits")]
arc_datasets = [("S20230513S0011_slit.fits", {"processed_bias": "S20230513S0012_slit_bias.fits",
                                              "processed_slitflat": "S20230513S0010_slit_slitflat.fits"})]
sci_datasets = [("S20230513S0232_slit.fits", {"processed_bias": "S20230513S0012_slit_bias.fits",
                                              "processed_slitflat": "S20230513S0010_slit_slitflat.fits"})]


@pytest.mark.integration_test
@pytest.mark.ghostslit
@pytest.mark.parametrize("input_filename", bias_datasets)
def test_reduce_slit_bias(input_filename, path_to_inputs, path_to_refs, change_working_dir):
    """Test the complete reduction of slitviewer bias frames"""
    ad = astrodata.open(os.path.join(path_to_inputs, input_filename))
    p = GHOSTSlit([ad])
    with change_working_dir():
        makeProcessedSlitBias(p)
        output_filename = p.streams['main'][0].filename
        adout = astrodata.open(os.path.join("calibrations", "processed_bias", output_filename))
        adref = astrodata.open(os.path.join(path_to_refs, output_filename))
        assert ad_compare(adref, adout, ignore_kw=['PROCBIAS', 'OVERRDNS', 'RDNOISEM'])


@pytest.mark.integration_test
@pytest.mark.ghostslit
@pytest.mark.parametrize("input_filename, processed_bias", flat_datasets)
def test_reduce_slit_flat(input_filename, processed_bias, path_to_inputs,
                          path_to_refs, change_working_dir):
    """Test the complete reduction of slitviewer flat frames"""
    ad = astrodata.open(os.path.join(path_to_inputs, input_filename))
    processed_bias = os.path.join(path_to_inputs, processed_bias)
    ucals = {"processed_bias": processed_bias}
    p = GHOSTSlit([ad], ucals=ucals)
    with change_working_dir():
        makeProcessedSlitFlat(p)
        output_filename = p.streams['main'][0].filename
        adout = astrodata.open(os.path.join("calibrations", "processed_slitflat", output_filename))
        adref = astrodata.open(os.path.join(path_to_refs, output_filename))
        assert ad_compare(adref, adout, ignore_kw=['PRSLITFL', 'OVERRDNS', 'RDNOISEM'])


@pytest.mark.integration_test
@pytest.mark.ghostslit
@pytest.mark.parametrize("input_filename, caldict", arc_datasets)
def test_reduce_slit_arc(input_filename, caldict, path_to_inputs,
                         path_to_refs, change_working_dir):
    """Test the complete reduction of slitviewer arc frames"""
    ad = astrodata.open(os.path.join(path_to_inputs, input_filename))
    ucals = {k: os.path.join(path_to_inputs, v)
             for k, v in caldict.items()}
    p = GHOSTSlit([ad], ucals=ucals)
    with change_working_dir():
        makeProcessedSlitArc(p)
        output_filename = p.streams['main'][0].filename
        adout = astrodata.open(os.path.join("calibrations", "processed_slit", output_filename))
        adref = astrodata.open(os.path.join(path_to_refs, output_filename))
        assert ad_compare(adref, adout, ignore_kw=['PRSLITIM', 'OVERRDNS', 'RDNOISEM'])


@pytest.mark.integration_test
@pytest.mark.ghostslit
@pytest.mark.parametrize("input_filename, caldict", sci_datasets)
def test_reduce_slit_science(input_filename, caldict, path_to_inputs,
                             path_to_refs, change_working_dir):
    """Test the complete reduction of slitviewer science frames"""
    ad = astrodata.open(os.path.join(path_to_inputs, input_filename))
    ucals = {k: os.path.join(path_to_inputs, v)
             for k, v in caldict.items()}
    p = GHOSTSlit([ad], ucals=ucals)
    with change_working_dir():
        makeProcessedSlit(p)
        for output_filename in [ad.filename for ad in p.streams['main']]:
            adout = astrodata.open(os.path.join("calibrations", "processed_slit", output_filename))
            adref = astrodata.open(os.path.join(path_to_refs, output_filename))
            assert ad_compare(adref, adout, ignore_kw=['PRSLITIM', 'OVERRDNS', 'RDNOISEM'])
