# import os
# from copy import deepcopy

import os

import astrodata
import gemini_instruments
import numpy as np
import pytest
from astrodata.testing import download_from_archive
from geminidr.core.primitives_preprocess import Preprocess
from geminidr.gemini.lookups import DQ_definitions as DQ
# from geminidr.gmos.primitives_gmos_image import GMOSImage
from geminidr.niri.primitives_niri_image import NIRIImage
from gempy.library.astrotools import cartesian_regions_to_slices
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_array_equal)

DEBUG = bool(os.getenv('DEBUG', False))


@pytest.fixture
def niriprim():
    file_path = download_from_archive("N20190120S0287.fits")
    ad = astrodata.open(file_path)
    p = NIRIImage([ad])
    p.addDQ()
    return p


@pytest.fixture
def niriprim2():
    file_path = download_from_archive("N20190120S0287.fits")
    ad = astrodata.open(file_path)
    ad.append(ad[0])
    p = NIRIImage([ad])
    p.addDQ()
    return p


@pytest.mark.dragons_remote_data
def test_apply_dq_plane_default(niriprim):
    """Default params: replace masked pixels by median of the image."""
    ad = niriprim.applyDQPlane()[0]
    assert_array_equal(ad[0].data[527:533, 430:435], 35)


@pytest.mark.dragons_remote_data
def test_apply_dq_plane_fixed_value(niriprim):
    """Replace by fixed value"""
    ad = niriprim.applyDQPlane(replace_value=0)[0]
    assert_array_equal(ad[0].data[527:533, 430:435], 0)


@pytest.mark.dragons_remote_data
def test_apply_dq_plane_mean(niriprim):
    """Replace by mean."""
    ad = niriprim.applyDQPlane(replace_value="mean")[0]
    assert_array_almost_equal(ad[0].data[527:533, 430:435], 38.56323,
                              decimal=5)


@pytest.mark.dragons_remote_data
def test_apply_dq_plane_replace_flags(niriprim):
    """Replace only given flags."""

    # First check with no_data, which is not present in the mask so pixel
    # should not be changed
    data_orig = niriprim.streams['main'][0][0].data.copy()
    ad = niriprim.applyDQPlane(replace_flags=DQ.no_data, replace_value=0)[0]
    assert_array_equal(ad[0].data[527:533, 430:435],
                       data_orig[527:533, 430:435])

    # Now with bad_pixel, so we should get 0 for this region
    ad = niriprim.applyDQPlane(replace_flags=DQ.bad_pixel, replace_value=0)[0]
    assert_array_equal(ad[0].data[527:533, 430:435], 0)


@pytest.mark.dragons_remote_data
def test_apply_dq_plane_ring_median(niriprim):
    """Replace by fixed value"""
    ad = niriprim.applyDQPlane(replace_value='median', inner=3, outer=5)[0]
    assert_array_equal(ad[0].data[527:533, 430:435],
                       [[26., 27., 27., 27., 29.],
                        [25., 27., 27., 26., 26.],
                        [26., 27., 26., 25., 26.],
                        [31., 31., 29., 25., 29.],
                        [31., 30., 27., 26., 27.],
                        [31., 30., 27., 26., 28.]])


@pytest.mark.dragons_remote_data
def test_apply_dq_plane_ring_mean(niriprim):
    """Replace by fixed value"""
    ad = niriprim.applyDQPlane(replace_value='mean', inner=3, outer=5)[0]
    assert_array_almost_equal(
        ad[0].data[527:533, 430:435],
        [[28.428572, 28.82353, 44.878788, 43.6, 43.314285],
         [27.6, 28.32353, 45.14706, 45.45714, 31.17647],
         [27.710526, 28.846153, 42.71795, 42.75, 30.868422],
         [41.871796, 43.825, 44.1, 42.325, 30.710526],
         [44., 45.675674, 48.166668, 32.555557, 30.694445],
         [31.68421, 32.432434, 33.7027, 32.675674, 30.552631]],
        decimal=5
    )


@pytest.mark.dragons_remote_data
def test_fixpixels(niriprim):
    regions = [
        '430:437,513:533',  # vertical region
        '450,521',  # single pixel
        '429:439,136:140',  # horizontal region
    ]
    ad = niriprim.fixPixels(regions=';'.join(regions), debug=DEBUG)[0]

    sy, sx = cartesian_regions_to_slices(regions[0])
    assert_almost_equal(ad[0].data[sy, sx].min(), 18.555, decimal=2)
    assert_almost_equal(ad[0].data[sy, sx].max(), 42.888, decimal=2)

    sy, sx = cartesian_regions_to_slices(regions[1])
    assert_almost_equal(ad[0].data[sy, sx].min(), 24.5, decimal=2)
    assert_almost_equal(ad[0].data[sy, sx].max(), 24.5, decimal=2)

    sy, sx = cartesian_regions_to_slices(regions[2])
    assert_almost_equal(ad[0].data[sy, sx].min(), 37.166, decimal=2)
    assert_almost_equal(ad[0].data[sy, sx].max(), 60.333, decimal=2)


@pytest.mark.dragons_remote_data
def test_fixpixels_median(niriprim):
    regions = [
        '450,521',  # single pixel
    ]
    ad = niriprim.fixPixels(regions=';'.join(regions),
                            use_local_median=True, debug=DEBUG)[0]

    sy, sx = cartesian_regions_to_slices(regions[0])
    assert_almost_equal(ad[0].data[sy, sx].min(), 28, decimal=2)
    assert_almost_equal(ad[0].data[sy, sx].max(), 28, decimal=2)


@pytest.mark.dragons_remote_data
def test_fixpixels_specify_axis(niriprim):
    regions = [
        '430:437,513:533',  # vertical region
    ]

    with pytest.raises(ValueError):
        ad = niriprim.fixPixels(regions=';'.join(regions), axis=0)[0]

    with pytest.raises(ValueError):
        ad = niriprim.fixPixels(regions=';'.join(regions), axis=3)[0]

    ad = niriprim.fixPixels(regions=';'.join(regions), axis=2, debug=DEBUG)[0]

    sy, sx = cartesian_regions_to_slices(regions[0])
    assert_almost_equal(ad[0].data[sy, sx].min(), 17.636, decimal=2)
    assert_almost_equal(ad[0].data[sy, sx].max(), 38.863, decimal=2)


@pytest.mark.dragons_remote_data
def test_fixpixels_with_file(niriprim, tmp_path):
    regions = [
        '450,521',  # single pixel
        '429:439,136:140',  # horizontal region
    ]
    regions_file = str(tmp_path / 'regions.txt')
    with open(regions_file, mode='w') as f:
        f.write('\n'.join(regions))

    ad = niriprim.fixPixels(regions='430:437,513:533',  # vertical region
                            regions_file=regions_file,
                            debug=DEBUG)[0]

    sy, sx = cartesian_regions_to_slices('430:437,513:533')
    assert_almost_equal(ad[0].data[sy, sx].min(), 18.555, decimal=2)
    assert_almost_equal(ad[0].data[sy, sx].max(), 42.888, decimal=2)

    sy, sx = cartesian_regions_to_slices(regions[0])
    assert_almost_equal(ad[0].data[sy, sx].min(), 24.5, decimal=2)
    assert_almost_equal(ad[0].data[sy, sx].max(), 24.5, decimal=2)

    sy, sx = cartesian_regions_to_slices(regions[1])
    assert_almost_equal(ad[0].data[sy, sx].min(), 37.166, decimal=2)
    assert_almost_equal(ad[0].data[sy, sx].max(), 60.333, decimal=2)


@pytest.mark.dragons_remote_data
def test_fixpixels_3D(astrofaker):
    np.random.seed(42)
    arr = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)

    # Shuffle the values to be sure the interpolation is done on the good axis
    # (when chacking the data below)
    i, k = np.arange(4), np.arange(6)
    np.random.shuffle(i)
    np.random.shuffle(k)
    arr = arr[i, :, :]
    arr = arr[:, :, k]
    refarr = arr.copy()

    # Set to 0 the region to be fixed
    arr[1:3, 2:4, 1:5] = 0

    ad = astrofaker.create('NIRI', 'IMAGE')
    ad.append(arr)
    p = Preprocess([ad])

    regions = ['2:5,3:4,2:3']
    ad = p.fixPixels(regions=';'.join(regions), debug=DEBUG)[0]

    assert_array_equal(refarr, ad[0].data)


@pytest.mark.dragons_remote_data
def test_fixpixels_3D_axis(astrofaker):
    np.random.seed(42)
    arr = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)

    # Shuffle the values to be sure the interpolation is done on the good axis
    # (when chacking the data below)
    j, k = np.arange(4), np.arange(6)
    np.random.shuffle(j)
    np.random.shuffle(k)
    arr = arr[:, j, :]
    arr = arr[:, :, k]
    refarr = arr.copy()

    # Set to 0 the region to be fixed
    arr[1:3, 2:4, 1:5] = 0

    ad = astrofaker.create('NIRI', 'IMAGE')
    ad.append(arr)
    p = Preprocess([ad])

    regions = ['2:5,3:4,2:3']
    ad = p.fixPixels(regions=';'.join(regions), debug=DEBUG, axis=3)[0]

    assert_array_equal(refarr, ad[0].data)


@pytest.mark.dragons_remote_data
def test_fixpixels_multiple_ext(niriprim2):
    regions = [
        '430:437,513:533',  # vertical region
        '1/450,521',  # single pixel
        '2/429:439,136:140',  # horizontal region
    ]
    ad = niriprim2.fixPixels(regions=';'.join(regions), debug=DEBUG)[0]

    # for all extensions
    sy, sx = cartesian_regions_to_slices(regions[0])
    assert_almost_equal(ad[0].data[sy, sx].min(), 18.555, decimal=2)
    assert_almost_equal(ad[0].data[sy, sx].max(), 42.888, decimal=2)
    assert_almost_equal(ad[1].data[sy, sx].min(), 18.555, decimal=2)
    assert_almost_equal(ad[1].data[sy, sx].max(), 42.888, decimal=2)

    # only ext 1
    sy, sx = cartesian_regions_to_slices(regions[1][2:])
    assert_almost_equal(ad[0].data[sy, sx].min(), 24.5, decimal=2)
    assert_almost_equal(ad[0].data[sy, sx].max(), 24.5, decimal=2)
    assert_almost_equal(ad[1].data[sy, sx].min(), 2733, decimal=2)
    assert_almost_equal(ad[1].data[sy, sx].max(), 2733, decimal=2)

    # only ext 2
    sy, sx = cartesian_regions_to_slices(regions[2][2:])
    assert_almost_equal(ad[0].data[sy, sx].min(), -125, decimal=2)
    assert_almost_equal(ad[0].data[sy, sx].max(), 21293, decimal=2)
    assert_almost_equal(ad[1].data[sy, sx].min(), 37.166, decimal=2)
    assert_almost_equal(ad[1].data[sy, sx].max(), 60.333, decimal=2)


# TODO @bquint: clean up these tests

# @pytest.fixture
# def niri_images(astrofaker):
#     """Create two NIRI images, one all 1s, the other all 2s"""
#     adinputs = []
#     for i in (1, 2):
#         ad = astrofaker.create('NIRI', 'IMAGE')
#         ad.init_default_extensions()
#         ad[0].data += i

#     adinputs.append(ad)

#     return NIRIImage(adinputs)


# @pytest.mark.xfail(reason="Test needs revision", run=False)
# def test_scale_by_exposure_time(niri_images):
#     ad1, ad2 = niri_images.streams['main']

#     ad2.phu[ad2._keyword_for('exposure_time')] *= 0.5
#     ad2_orig_value = ad2[0].data.mean()

#     ad1, ad2 = niri_images.scaleByExposureTime(time=None)

#     # Check that ad2 had its data doubled
#     assert abs(ad2[0].data.mean() - ad2_orig_value * 2) < 0.001

#     ad1, ad2 = niri_images.scaleByExposureTime(time=1)

#     # Check that ad2 has been rescaled to 1-second
#     print(ad2[0].data.mean(), ad2_orig_value, ad2.phu["ORIGTEXP"])
#     assert abs(ad2[0].data.mean() - ad2_orig_value / ad2.phu["ORIGTEXP"]) < 0.001


# @pytest.mark.xfail(reason="Test needs revision", run=False)
# def test_add_object_mask_to_dq(astrofaker):
#     ad_orig = astrofaker.create('F2', 'IMAGE')

#     # astrodata.open(os.path.join(TESTDATAPATH, 'GMOS', 'N20150624S0106_refcatAdded.fits'))
#     p = GMOSImage([deepcopy(ad_orig)])
#     ad = p.addObjectMaskToDQ()[0]

#     for ext, ext_orig in zip(ad, ad_orig):
#         assert all(ext.mask[ext.OBJMASK == 0] == ext_orig.mask[ext.OBJMASK == 0])
#         assert all(ext.mask[ext.OBJMASK == 1] == ext_orig.mask[ext.OBJMASK == 1] | 1)


# @pytest.mark.xfail(reason="Test needs revision", run=False)
# def test_adu_to_electrons(astrofaker):
#     ad = astrofaker.create("NIRI", "IMAGE")
#     # astrodata.open(os.path.join(TESTDATAPATH, 'NIRI', 'N20070819S0104_dqAdded.fits'))
#     p = NIRIImage([ad])
#     ad = p.ADUToElectrons()[0]
#     assert ad_compare(ad, os.path.join(TESTDATAPATH, 'NIRI',
#                                        'N20070819S0104_ADUToElectrons.fits'))


# @pytest.mark.xfail(reason="Test needs revision", run=False)
# def test_associateSky():
#     filenames = ['N20070819S{:04d}_flatCorrected.fits'.format(i)
#                  for i in range(104, 109)]

#     adinputs = [astrodata.open(os.path.join(TESTDATAPATH, 'NIRI', f))
#                 for f in filenames]

#     p = NIRIImage(adinputs)
#     p.separateSky()  # Difficult to construct this by hand
#     p.associateSky()
#     filename_set = {ad.phu['ORIGNAME'] for ad in adinputs}

#     # Test here is that each science frame has all other frames as skies
#     for k, v in p.sky_dict.items():
#         v = [ad.phu['ORIGNAME'] for ad in v]
#         assert len(v) == len(filenames) - 1
#         assert set([k] + v) == filename_set

# def test_correctBackgroundToReference(self):
#     pass

# def test_darkCorrect(self):
#     ad = astrodata.open(os.path.join(TESTDATAPATH, 'NIRI',
#                             'N20070819S0104_nonlinearityCorrected.fits'))
#     p = NIRIImage([ad])
#     ad = p.darkCorrect()[0]
#     assert ad_compare(ad, os.path.join(TESTDATAPATH, 'NIRI',
#                             'N20070819S0104_darkCorrected.fits'))

# def test_darkCorrect_with_af(self):
#     science = astrofaker.create('NIRI', 'IMAGE')
#     dark = astrofaker.create('NIRI', 'IMAGE')
#     p = NIRIImage([science])
#     p.darkCorrect([science], dark=dark)
#     science.subtract(dark)
#     assert ad_compare(science, dark)


# af.init_default_extensions()
# af[0].mask = np.zeros_like(af[0].data, dtype=np.uint16)
# def test_flatCorrect(self):
#     ad = astrodata.open(os.path.join(TESTDATAPATH, 'NIRI',
#                             'N20070819S0104_darkCorrected.fits'))
#     p = NIRIImage([ad])
#     ad = p.flatCorrect()[0]
#     assert ad_compare(ad, os.path.join(TESTDATAPATH, 'NIRI',
#                             'N20070819S0104_flatCorrected.fits'))
#
# def test_makeSky(self):
#     pass
#
# def test_nonlinearityCorrect(self):
#     # Don't use NIRI data; NIRI has its own primitive
#     pass
#
# def test_normalizeFlat(self):
#     flat_file = os.path.join(TESTDATAPATH, 'NIRI',
#                             'N20070913S0220_flat.fits')
#     ad = astrodata.open(flat_file)
#     ad.multiply(10.0)
#     del ad.phu['NORMLIZE']  # Delete timestamp of previous processing
#     p = NIRIImage([ad])
#     ad = p.normalizeFlat(suffix='_flat', strip=True)[0]
#     assert ad_compare(ad, flat_file)
#
# def test_separateSky(self):
#     pass
#
# def test_skyCorrect(self):
#     pass
#
# def test_subtractSky(self):
#     pass
#
# def test_subtractSkyBackground(self):
#     ad = astrodata.open(os.path.join(TESTDATAPATH, 'NIRI',
#                             'N20070819S0104_flatCorrected.fits'))
#     ad.hdr['SKYLEVEL'] = 1000.0
#     orig_data = ad[0].data.copy()
#     p = NIRIImage([ad])
#     ad = p.subtractSkyBackground()[0]
#     assert (orig_data - ad[0].data).min() > 999.99
#     assert (orig_data - ad[0].data).max() < 1000.01
#
# def test_thresholdFlatfield(self):
#     ad = astrodata.open(os.path.join(TESTDATAPATH, 'NIRI',
#                                      'N20070913S0220_flat.fits'))
#     del ad.phu['TRHFLAT']  # Delete timestamp of previous processing
#     ad[0].data[100, 100] = 20.0
#     p = NIRIImage([ad])
#     ad = p.thresholdFlatfield()[0]
#     assert ad[0].mask[100, 100] == 64
