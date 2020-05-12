#!/usr/bin/env python
import os
import numpy as np
import pytest

import astrodata
# noinspection PyUnresolvedReferences
import gemini_instruments
from astrodata import testing as ad_testing
from gempy.utils import logutils
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals

# noinspection PyUnresolvedReferences
from recipe_system.testing import reduce_bias, reference_ad

datasets = [
    "S20180707S0043.fits",  # B600 at 0.520 um
    "S20190502S0096.fits",  # B600 at 0.525 um
    "S20200122S0020.fits",  # B600 at 0.520 um
    "N20200101S0055.fits",  # B1200 at 0.495 um
    # "S20180410S0120.fits",  # B1200 at 0.595 um  # Scattered light?
    # "S20190410S0053.fits",  # B1200 at 0.463 um  # Scattered light?
]


# -- Tests --------------------------------------------------------------------
# @pytest.mark.skip(reason="Arrays are not almost equal to 3 decimals")
@pytest.mark.gmosls
def test_processed_flat_has_median_around_one(processed_flat):
    for ext in processed_flat:
        data = np.ma.masked_array(ext.data, mask=ext.mask)
        np.testing.assert_almost_equal(np.median(data.ravel()), 1.0, decimal=3)


# @pytest.mark.skip(reason="High std")
@pytest.mark.gmosls
def test_processed_flat_has_small_std(processed_flat):
    for ext in processed_flat:
        data = np.ma.masked_array(ext.data, mask=ext.mask)
        np.testing.assert_array_less(np.std(data.ravel()), 0.1)


#@pytest.mark.skip(reason='ref data needs to be updated')
@pytest.mark.gmosls
def test_regression_processed_flat(processed_flat, reference_ad):
    ref_flat = reference_ad(processed_flat.filename)
    for ext, ext_ref in zip(processed_flat, ref_flat):
        np.testing.assert_allclose(ext.mask, ext_ref.mask)
        np.testing.assert_almost_equal(ext.data, ext_ref.data, decimal=3)


# -- Fixtures ----------------------------------------------------------------
@pytest.fixture(scope='module', params=datasets)
def processed_flat(
        cache_file_from_archive, change_working_dir, reduce_bias, request):

    filename = request.param
    path = cache_file_from_archive(filename)
    ad = astrodata.open(path)

    cals = ad_testing.get_associated_calibrations(filename)
    cals = cals[cals.caltype.str.contains('bias')]
    cals = [cache_file_from_archive(c) for c in cals.filename.values]

    master_bias = reduce_bias(ad.data_label(), cals)

    with change_working_dir():
        print("Reducing FLATs in folder:\n  {}".format(os.getcwd()))
        logutils.config(file_name='log_flat_{}.txt'.format(ad.data_label()))

        calibration_files = ['processed_bias:{}'.format(master_bias)]

        reduce = Reduce()
        reduce.files.extend([path])
        reduce.mode = 'ql'
        reduce.ucals = normalize_ucals(reduce.files, calibration_files)
        reduce.runr()

        master_flat = reduce.output_filenames.pop()
        master_flat_ad = astrodata.open(master_flat)

    return master_flat_ad


if __name__ == '__main__':
    pytest.main()
