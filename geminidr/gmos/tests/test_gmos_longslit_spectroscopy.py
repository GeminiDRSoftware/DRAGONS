#!/usr/bin/python
"""
Tests related to GMOS Long-slit Spectroscopy data reduction.

"""
import glob
import pytest
import os

import astrodata
import gemini_instruments
import geminidr

from geminidr.gmos import primitives_gmos_spect
from gempy.adlibrary import dataselect
from gempy.utils import logutils
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals


@pytest.fixture(scope='module')
def calibrations():

    my_cals = []

    return my_cals


def test_can_run_reduce_bias(path_to_inputs, calibrations):
    """
    Make sure that the reduce_BIAS works for spectroscopic data.
    """

    raw_subdir = 'GMOS/GN-2017A-FT-19'

    logutils.config(file_name='reduce_GMOS_LS_bias.log')

    all_files = sorted(glob.glob(os.path.join(path_to_inputs, raw_subdir, '*.fits')))
    assert len(all_files) > 1

    list_of_bias = dataselect.select_data(all_files, ['BIAS'], [])

    reduce_bias = Reduce()
    assert len(reduce_bias.files) == 0

    reduce_bias.files.extend(list_of_bias)
    assert len(reduce_bias.files) == len(list_of_bias)

    reduce_bias.runr()

    calibrations.append(
        'processed_bias:{}'.format(reduce_bias.output_filenames[0])
    )


def test_can_run_reduce_flat(path_to_inputs, calibrations):
    """
    Make sure that the reduce_FLAT_LS_SPECT works for spectroscopic data.
    """

    raw_subdir = 'GMOS/GN-2017A-FT-19'

    logutils.config(file_name='reduce_GMOS_LS_flat.log')

    assert len(calibrations) == 1

    all_files = sorted(glob.glob(os.path.join(path_to_inputs, raw_subdir, '*.fits')))
    assert len(all_files) > 1

    list_of_flat = dataselect.select_data(all_files, ['FLAT'], [])

    reduce_flat = Reduce()
    assert len(reduce_flat.files) == 0

    reduce_flat.files.extend(list_of_flat)
    assert len(reduce_flat.files) == len(list_of_flat)

    reduce_flat.ucals = normalize_ucals(reduce_flat.files, calibrations)

    reduce_flat.runr()

    # calibrations.append(
    #     'processed_flat:{}'.format(reduce_flat.output_filenames[0])
    # )


def test_can_run_reduce_arc(path_to_inputs, calibrations):
    """
    Make sure that the reduce_FLAT_LS_SPECT works for spectroscopic
    data.
    """

    raw_subdir = 'GMOS/GN-2017A-FT-19'

    logutils.config(file_name='reduce_GMOS_LS_arc.log')

    all_files = sorted(glob.glob(os.path.join(path_to_inputs, raw_subdir, '*.fits')))
    assert len(all_files) > 1

    list_of_arcs = dataselect.select_data(all_files, ['ARC'], [])

    for f in list_of_arcs:
        ad = astrodata.open(f)
        _ = ad.gain_setting()

    for c in calibrations:
        f = c.split(':')[-1]
        ad = astrodata.open(f)
        _ = ad.gain_setting()

    temp = [c for c in calibrations if 'bias' in c]
    processed_bias = temp[0].split(':')[-1]

    adinputs = [astrodata.open(f) for f in list_of_arcs]

    p = primitives_gmos_spect.GMOSSpect(adinputs)

    p.viewer = geminidr.dormantViewer(p, None)

    p.prepare()
    p.addDQ(static_bpm=None)
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect(bias=processed_bias)
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.mosaicDetectors()
    p.makeIRAFCompatible()
    p.writeOutputs()  # for now, to speed up diagnostics of the next step
    p.determineWavelengthSolution()
    p.determineDistortion()
    p.storeProcessedArc()
    p.writeOutputs()


# ToDo WIP - Define first how flats are processed
# def test_can_run_reduce_science(path_to_inputs, calibrations):
#     """
#     Make sure that the recipes_ARC_LS_SPECT works for spectroscopic data.
#     """

    # raw_subdir = 'GMOS/GN-2017A-FT-19'
    #
    # logutils.config(file_name='reduce_GMOS_LS_arc.log')
    #
    # assert len(calibrations) == 2
    #
    # all_files = sorted(glob.glob(os.path.join(path_to_inputs, raw_subdir, '*.fits')))
    # assert len(all_files) > 1
    #
    # list_of_science = dataselect.select_data(all_files, [], ['CAL'])
    #
    # reduce_science = Reduce()
    # assert len(reduce_science.files) == 0
    #
    # reduce_science.files.extend(list_of_science)
    # assert len(reduce_science.files) == len(list_of_science)
    #
    # reduce_science.ucals = normalize_ucals(reduce_science.files, calibrations)
    #
    # reduce_science.runr()


if __name__ == '__main__':
    pytest.main()
