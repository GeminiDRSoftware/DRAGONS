#!/usr/bin/env python

import glob
import os

import pytest

from astrodata.testing import download_from_archive
from gempy.adlibrary import dataselect
from gempy.utils import logutils
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals

datasets = [

    # Darks 3s
    "S20131127S0257.fits",
    "S20131127S0258.fits",
    "S20131127S0259.fits",
    # "S20131127S0260.fits",
    # "S20131127S0261.fits",
    # "S20131127S0262.fits",
    # "S20131127S0263.fits",

    # Darks 20s
    "S20130930S0242.fits",
    "S20130930S0243.fits",
    "S20130930S0244.fits",
    # "S20130930S0245.fits",
    # "S20130930S0246.fits",
    # "S20131023S0193.fits",
    # "S20131023S0194.fits",
    # "S20131023S0195.fits",
    # "S20131023S0196.fits",
    # "S20131023S0197.fits",
    # "S20140124S0033.fits",
    # "S20140124S0034.fits",
    # "S20140124S0035.fits",
    # "S20140124S0036.fits",
    # "S20140124S0037.fits",
    # "S20140124S0038.fits",
    # "S20140209S0542.fits",
    # "S20140209S0543.fits",
    # "S20140209S0544.fits",
    # "S20140209S0545.fits",

    # Darks 120s
    "S20131121S0115.fits",
    "S20131120S0116.fits",
    "S20131120S0117.fits",
    # "S20131120S0118.fits",
    # "S20131120S0119.fits",
    # "S20131120S0120.fits",
    # "S20131121S0010.fits",
    # "S20131122S0012.fits",
    # "S20131122S0438.fits",
    # "S20131122S0439.fits",

    # Flats Y
    "S20131126S1111.fits",
    "S20131126S1112.fits",
    "S20131126S1113.fits",
    # "S20131126S1114.fits",
    # "S20131126S1115.fits",
    # "S20131126S1116.fits",
    # "S20131129S0320.fits",
    # "S20131129S0321.fits",
    # "S20131129S0322.fits",
    # "S20131129S0323.fits",

    # Science images
    "S20131121S0075.fits",
    "S20131121S0076.fits",
    "S20131121S0077.fits",
    # "S20131121S0078.fits",
    # "S20131121S0079.fits",
    # "S20131121S0080.fits",
    # "S20131121S0081.fits",
    # "S20131121S0082.fits",
    # "S20131121S0083.fits",

]


@pytest.mark.skip(reason='Test needs refactoring')
@pytest.mark.f2image
@pytest.mark.integration_test
def test_reduce_image(change_working_dir):
    with change_working_dir():
        calib_files = []
        all_files = [download_from_archive(f) for f in datasets]
        all_files.sort()
        assert len(all_files) > 1

        darks_3s = dataselect.select_data(
            all_files, ['F2', 'DARK', 'RAW'], [],
            dataselect.expr_parser('exposure_time==3'))
        darks_3s.sort()

        darks_20s = dataselect.select_data(
            all_files, ['F2', 'DARK', 'RAW'], [],
            dataselect.expr_parser('exposure_time==20'))
        darks_20s.sort()

        darks_120s = dataselect.select_data(
            all_files, ['F2', 'DARK', 'RAW'], [],
            dataselect.expr_parser('exposure_time==120'))
        darks_120s.sort()

        flats = dataselect.select_data(
            all_files, ['F2', 'FLAT', 'RAW'], [],
            dataselect.expr_parser('filter_name=="Y"'))
        flats.sort()

        science = dataselect.select_data(
            all_files, ['F2', 'RAW'], ['CAL'],
            dataselect.expr_parser('filter_name=="Y"'))
        science.sort()

        for darks in [darks_3s, darks_20s, darks_120s]:
            reduce_darks = Reduce()
            assert len(reduce_darks.files) == 0

            reduce_darks.files.extend(darks)
            assert len(reduce_darks.files) == len(darks)

            logutils.config(file_name='f2_test_reduce_darks.log', mode='quiet')
            reduce_darks.runr()

            calib_files.append(
                'processed_dark:{}'.format(reduce_darks.output_filenames[0])
            )

        logutils.config(file_name='f2_test_reduce_bpm.log', mode='quiet')
        reduce_bpm = Reduce()
        reduce_bpm.files.extend(flats)
        assert len(reduce_bpm.files) == len(flats)

        reduce_bpm.files.extend(darks_3s)
        assert len(reduce_bpm.files) == len(flats) + len(darks_3s)

        reduce_bpm.recipename = 'makeProcessedBPM'
        reduce_bpm.runr()

        bpm_filename = reduce_bpm.output_filenames[0]

        logutils.config(file_name='f2_test_reduce_flats.log', mode='quiet')
        reduce_flats = Reduce()
        reduce_flats.files.extend(flats)
        reduce_flats.uparms = [('addDQ:user_bpm', bpm_filename)]
        reduce_flats.runr()

        calib_files.append(
            'processed_flat:{}'.format(reduce_flats.output_filenames[0])
        )

        logutils.config(file_name='f2_test_reduce_science.log', mode='quiet')
        reduce_target = Reduce()
        reduce_target.files.extend(science)
        reduce_target.uparms = [('addDQ:user_bpm', bpm_filename)]
        reduce_target.ucals = normalize_ucals(calib_files)
        reduce_target.runr()


if __name__ == '__main__':
    pytest.main()
