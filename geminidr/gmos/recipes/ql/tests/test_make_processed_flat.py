#!/usr/bin/env python
import numpy as np
import pytest

import astrodata
# noinspection PyUnresolvedReferences
import gemini_instruments
from astrodata import testing as ad_testing
from gempy.adlibrary import dataselect
# noinspection PyUnresolvedReferences
from recipe_system.testing import reduce_bias, reduce_flat, reference_ad

datasets = [
    "S20180707S0043.fits",  # B600 at 0.520 um
    # "S20190502S0096.fits",  # B600 at 0.525 um
    # "S20200122S0020.fits",  # B600 at 0.520 um
    # "N20200101S0055.fits",  # B1200 at 0.495 um
    # "S20180410S0120.fits",  # B1200 at 0.595 um  # Scattered light?
    # "S20190410S0053.fits",  # B1200 at 0.463 um  # Scattered light?
]


# -- Tests --------------------------------------------------------------------
@pytest.mark.skip(reason="Arrays are not almost equal to 3 decimals")
@pytest.mark.gmosls
@pytest.mark.parametrize("processed_flat", datasets, indirect=True)
def test_processed_flat_has_median_around_one(processed_flat):
    for ext in processed_flat:
        data = np.ma.masked_array(ext.data, mask=ext.mask)
        np.testing.assert_almost_equal(np.median(data.ravel()), 1.0, decimal=3)


@pytest.mark.skip(reason="High std")
@pytest.mark.gmosls
@pytest.mark.parametrize("processed_flat", datasets, indirect=True)
def test_processed_flat_has_small_std(processed_flat):
    for ext in processed_flat:
        data = np.ma.masked_array(ext.data, mask=ext.mask)
        np.testing.assert_array_less(np.std(data.ravel()), 0.1)


@pytest.mark.gmosls
@pytest.mark.parametrize("processed_flat", datasets, indirect=True)
def test_regression_processed_flat(processed_flat, reference_ad):
    ref_flat = reference_ad(processed_flat.filename)
    for ext, ext_ref in zip(processed_flat, ref_flat):
        np.testing.assert_allclose(ext.data, ext_ref.data, rtol=1e-7)


# -- Fixtures ----------------------------------------------------------------
@pytest.fixture(scope='module')
def processed_flat(cache_file_from_archive, reduce_bias, reduce_flat, request):
    filename = request.param
    path = cache_file_from_archive(filename)
    ad = astrodata.open(path)

    cals = ad_testing.get_associated_calibrations(filename)
    cals = [cache_file_from_archive(c) for c in cals.filename.values]

    master_bias = reduce_bias(
        ad.data_label(), dataselect.select_data(cals, tags=['BIAS']))

    master_flat = reduce_flat(
        ad.data_label(), dataselect.select_data(cals, tags=['FLAT']), master_bias)

    return master_flat


if __name__ == '__main__':
    pytest.main()
