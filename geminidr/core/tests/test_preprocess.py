from itertools import count
import os

import astrodata, gemini_instruments
import numpy as np
import pytest
from astrodata.testing import ad_compare, download_from_archive
from geminidr.core.primitives_preprocess import Preprocess
from geminidr.gemini.lookups import DQ_definitions as DQ
from geminidr.gsaoi.primitives_gsaoi_image import GSAOIImage
from geminidr.niri.primitives_niri_image import NIRIImage
from gempy.library.astrotools import cartesian_regions_to_slices
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_array_equal)


DEBUG = bool(os.getenv('DEBUG', False))

nonlinearity_datasets = (
    ('S20190115S0073_dqAdded.fits', 'S20190115S0073_nonlinearityCorrected_adu.fits'),
    ('S20190115S0073_varAdded.fits', 'S20190115S0073_nonlinearityCorrected_electron.fits'),
)

# ---- Fixtures ----------------------------------------

@pytest.fixture
def niri_images(niri_image):
    """Create two NIRI images, one all 1s, the other all 2s"""
    adinputs = []
    for i in (1, 2):
        ad = niri_image()
        ad[0].data += i

        adinputs.append(ad)

    return NIRIImage(adinputs)

@pytest.fixture
def niriprim():
    file_path = download_from_archive("N20190120S0287.fits")
    ad = astrodata.open(file_path)
    p = NIRIImage([ad])
    p.addDQ(static_bpm=download_from_archive("bpm_20010317_niri_niri_11_full_1amp.fits"))
    return p


@pytest.fixture
def niriprim2():
    file_path = download_from_archive("N20190120S0287.fits")
    ad = astrodata.open(file_path)
    ad.append(ad[0])
    p = NIRIImage([ad])
    p.addDQ()
    return p

@pytest.fixture
def niri_image(astrofaker):
    """Create a fake NIRI image.

    Optional
    --------
    keywords : dict
        A dictionary with keys equal to FITS header keywords, whose values
        will be propogated to the new image.

    """

    def _niri_image(filename='N20010101S0001.fits', keywords={}):

        ad = astrofaker.create('NIRI', 'IMAGE',
                                extra_keywords=keywords,
                                filename=filename)
        ad.init_default_extensions()
        return ad

    return _niri_image

@pytest.fixture
def niri_sequence(niri_image):
    """Create a sequence of fake NIRI images.

    Parameters
    ----------
    marker : str, Options: ('object', 'sky1', 'sky2', 'sky3')

    Can be called in a test like `niri_sequence('object')` as long as it's
    passed as an argument.
    """

    # Use an infiite iterator to ensure fake files get unique filenames.
    filenum = count(1)

    def _niri_sequence(marker):

        nonlocal filenum

        adoutputs = []
        if marker == 'object':
            ra_offsets = [-6, -6, 0]
            dec_offsets = [0, -6, -6]
            guided = True
        elif marker == 'sky1':
            ra_offsets = [-180, -240, -240]
            dec_offsets = [180, 180, 120]
            guided = False
        elif marker == 'sky2':
            ra_offsets = [330, 330, 270]
            dec_offsets = [-280, -210, -280]
            guided = False
        elif marker == 'sky3':
            ra_offsets = [470, 430, 420]
            dec_offsets = [ 420, 420, 370]
            guided = False
        else:
            raise ValueError(f'"{marker}" not recognized as input')

        for raoff, decoff in zip(ra_offsets, dec_offsets):
            filename = f'N20010101S{next(filenum):04d}.fits'
            # Need to add unguided keywords to automatically detect as sky
            ad = niri_image(filename=filename)
            ad.sky_offset(raoff, decoff)
            if not guided:
                for keyword in ('PWFS1_ST', 'PWFS2_ST', 'OIWFS_ST'):
                    ad.phu[keyword] = 'True'
            adoutputs.append(ad)

        return adoutputs

    return _niri_sequence

# ---- Tests ---------------------------------------------

def test_adu_to_electrons(niri_image):
    ad = niri_image()
    ad[0].data += 1
    gain = ad.gain()[0]
    p = NIRIImage([ad])
    orig_sat = ad.saturation_level()[0]
    orig_nonlin = ad.non_linear_level()[0]
    p.prepare()
    assert ad.saturation_level()[0] == orig_sat
    assert ad.non_linear_level()[0] == orig_nonlin
    p.ADUToElectrons()
    assert ad.gain() == [1.0]
    assert ad.saturation_level()[0] == orig_sat * gain
    assert ad.non_linear_level()[0] == orig_nonlin * gain
    assert_array_almost_equal(ad[0].data, gain)


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

    for region in regions:
        sy, sx = cartesian_regions_to_slices(region)
        assert_array_equal(ad[0].mask[sy, sx] & DQ.no_data, DQ.no_data)

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
def test_fixpixels_errors(niriprim):
    with pytest.raises(ValueError, match="region .* out of bound"):
        niriprim.fixPixels(regions='4300,*')[0]

    with pytest.raises(ValueError,
                       match="no good data left for the interpolation"):
        niriprim.fixPixels(regions='*,*')[0]

    with pytest.raises(ValueError, match="no good data left for the "
                       "interpolation along the chosen axis"):
        niriprim.fixPixels(regions='430,*', axis=2)[0]


@pytest.mark.dragons_remote_data
def test_fixpixels_median(niriprim):
    regions = [
        '450,521',  # single pixel
    ]
    ad = niriprim.fixPixels(regions=';'.join(regions),
                            use_local_median=True, debug=DEBUG)[0]

    sy, sx = cartesian_regions_to_slices(regions[0])
    assert_array_equal(ad[0].mask[sy, sx] & DQ.no_data, DQ.no_data)
    assert_almost_equal(ad[0].data[sy, sx].min(), 28, decimal=2)
    assert_almost_equal(ad[0].data[sy, sx].max(), 28, decimal=2)


@pytest.mark.dragons_remote_data
def test_fixpixels_column(niriprim):
    regions = ['433,*']
    ad = niriprim.fixPixels(regions=';'.join(regions),
                            use_local_median=True, debug=DEBUG)[0]
    assert_almost_equal(ad[0].data[500:527, 432].min(), 18.5, decimal=2)
    assert_almost_equal(ad[0].data[500:527, 432].max(), 43, decimal=2)


@pytest.mark.dragons_remote_data
def test_fixpixels_line(niriprim):
    regions = ['*, 533']
    ad = niriprim.fixPixels(regions=';'.join(regions),
                            use_local_median=True, debug=DEBUG)[0]
    assert_almost_equal(ad[0].data[532, 430:435].min(), 22, decimal=2)
    assert_almost_equal(ad[0].data[532, 430:435].max(), 38.5, decimal=2)


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
    assert_array_equal(ad[0].mask[sy, sx] & DQ.no_data, DQ.no_data)
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
    arr = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)

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
    arr = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)

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
        '430:437, 513:533',  # vertical region
        '1 / 450,521',  # single pixel
        '2/429:439, 136:140',  # horizontal region
    ]
    ad = niriprim2.fixPixels(regions=';'.join(regions), debug=DEBUG)[0]

    # for all extensions
    sy, sx = cartesian_regions_to_slices(regions[0])
    assert_almost_equal(ad[0].data[sy, sx].min(), 18.555, decimal=2)
    assert_almost_equal(ad[0].data[sy, sx].max(), 42.888, decimal=2)
    assert_almost_equal(ad[1].data[sy, sx].min(), 18.555, decimal=2)
    assert_almost_equal(ad[1].data[sy, sx].max(), 42.888, decimal=2)

    # only ext 1
    sy, sx = cartesian_regions_to_slices(regions[1][3:])
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


@pytest.mark.regression
@pytest.mark.preprocessed_data
@pytest.mark.parametrize('dataset', nonlinearity_datasets)
#def test_nonlinearity_correct(path_to_inputs, path_to_refs, dataset):
def test_nonlinearity_correct(path_to_inputs, path_to_refs, dataset):
    """Only GSAOI uses the core primitive with real coefficients"""
    ad = astrodata.open(os.path.join(path_to_inputs, dataset[0]))
    p = GSAOIImage([ad])
    ad_out = p.nonlinearityCorrect().pop()
    ad_ref = astrodata.open(os.path.join(path_to_refs, dataset[1]))

    assert ad_compare(ad_out, ad_ref, ignore=['filename'])


# TODO @bquint: clean up these tests

def test_scale_by_exposure_time(niri_images):
    ad1, ad2 = niri_images.streams['main']

    ad2.phu[ad2._keyword_for('exposure_time')] *= 0.5
    ad2_orig_value = ad2[0].data.mean()

    ad1, ad2 = niri_images.scaleByExposureTime(time=None)

    # Check that ad2 had its data doubled
    assert abs(ad2[0].data.mean() - ad2_orig_value * 2) < 0.001

    ad1, ad2 = niri_images.scaleByExposureTime(time=1)

    # Check that ad2 has been rescaled to 1-second
    print(ad2[0].data.mean(), ad2_orig_value, ad2.phu["ORIGTEXP"])
    assert abs(ad2[0].data.mean() - ad2_orig_value / ad2.phu["ORIGTEXP"]) < 0.001


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


def test_associate_sky(niri_sequence):
    adinputs = niri_sequence('object')

    p = NIRIImage(adinputs)
    p.separateSky()  # Difficult to construct this by hand
    p.associateSky()
    filename_set = {ad.phu['ORIGNAME'] for ad in adinputs}

    # Test here is that each science frame has all other frames as skies
    for ad in p.showList():
        skies = [a[0].replace('_skyAssociated', '') for a in ad.SKYTABLE]
        assert len(ad.SKYTABLE) == len(niri_sequence('object')) - 1
        assert set([ad.phu['ORIGNAME']] + skies) == filename_set

def test_associate_sky_pass_skies(niri_sequence):
    obj_inputs = niri_sequence('object')
    sky_inputs = niri_sequence('sky1')

    in_sky_names = [ad.filename for ad in sky_inputs]

    p = NIRIImage(obj_inputs)
    # Don't run separateSky, this is to simulate starting and resuming work
    # with pre-known sky frames.
    p.associateSky(sky=sky_inputs)

    out_sky_names = [ad.phu['ORIGNAME'] for ad in p.streams['sky']]

    assert in_sky_names == out_sky_names

def test_associate_sky_use_all(niri_sequence):

    objects = niri_sequence('object')
    skies1 = niri_sequence('sky1')
    skies2 = niri_sequence('sky2')

    expected_skies = set([ad.filename for ad in skies2])

    p = NIRIImage(objects + skies1 + skies2)
    p.separateSky()
    # Check that 'use_all' sets all skies beyond the minimum distance as sky.
    # Skies from "sky1" should be within the minimum distance, so all frames
    # in the 'main' stream should have all skies from "sky2" in their SKYTABLE.
    p.associateSky(distance=305, use_all=True)

    for ad in p.streams['main']:
        skies = set([row[0].replace('_skyAssociated', '')
                     for row in ad.SKYTABLE])
        assert skies == expected_skies - set([ad.phu['ORIGNAME']])

def test_associate_sky_exclude_all(niri_sequence):

    objects = niri_sequence('object')
    skies1 = niri_sequence('sky1')

    p = NIRIImage(objects + skies1)
    p.separateSky()
    p.associateSky(distance=1000)

    # assert len(p.streams['no_skies']) == len(objects)

    for ad in p.showList():
        with pytest.raises(AttributeError):
            ad.SKYTABLE

def test_associate_sky_exclude_some(niri_image, niri_sequence):

    objects = niri_sequence('object')
    extra_frame = [niri_image(filename='N20010101S0099.fits')]
    extra_frame[0].sky_offset(600, 600)
    skies1 = niri_sequence('sky1')

    object_names = set([ad.filename for ad in objects])

    p = NIRIImage(objects + extra_frame + skies1)
    p.separateSky()
    p.associateSky(distance=500)

    no_skies = set([ad.phu['ORIGNAME'] for ad in p.streams['no_skies']])

    # Check that frames in 'objects' have been put in the 'no_skies' stream
    # since they're closer to the skies than the minimum distance
    assert object_names == no_skies


# def test_correctBackgroundToReference(self):
#     pass

# def test_darkCorrect(self):
#     ad = astrodata.open(os.path.join(TESTDATAPATH, 'NIRI',
#                             'N20070819S0104_nonlinearityCorrected.fits'))
#     p = NIRIImage([ad])
#     ad = p.darkCorrect()[0]
#     assert ad_compare(ad, os.path.join(TESTDATAPATH, 'NIRI',
#                             'N20070819S0104_darkCorrected.fits'))

@pytest.mark.xfail(reason="Test needs revision", run=False)
def test_darkCorrect_with_af(astrofaker):
    science = astrofaker.create('NIRI', 'IMAGE')
    dark = astrofaker.create('NIRI', 'IMAGE')
    p = NIRIImage([science])
    p.darkCorrect([science], dark=dark)
    science.subtract(dark)
    science.filename = 'N20010101S0001.fits'
    assert ad_compare(science, dark)


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

def test_separate_sky_offset(niri_sequence):

    object_frames = niri_sequence('object')
    sky_frames = niri_sequence('sky1')

    adinputs = object_frames + sky_frames

    target_filenames = set([ad.filename for ad in object_frames])
    sky_filenames = set([ad.filename for ad in sky_frames])

    p = NIRIImage(adinputs)
    p.separateSky()

    target_names = set([ad.phu['ORIGNAME'] for ad in p.streams['main']])
    sky_names = set([ad.phu['ORIGNAME'] for ad in p.streams['sky']])

    assert len(p.streams['main']) == len(object_frames)
    assert len(p.streams['sky']) == len(sky_frames)
    assert target_filenames == target_names
    assert sky_filenames == sky_names

@pytest.mark.parametrize('target', ['object', 'sky1'])
def test_separate_sky_all_one_type(target, niri_sequence):

    frames = niri_sequence(target)
    in_names = set([ad.filename for ad in frames])

    p = NIRIImage(frames)
    p.separateSky()

    out_obj_names = set(ad.phu['ORIGNAME'] for ad in p.streams['main'])
    out_sky_names = set(ad.phu['ORIGNAME'] for ad in p.streams['sky'])
    # Change to testing filenames
    assert out_obj_names == out_sky_names
    assert out_obj_names == in_names
    assert out_sky_names == in_names

@pytest.mark.parametrize('frac_FOV', [0.9, 0.5])
def test_separate_sky_frac_FOV(frac_FOV, niri_sequence):

    object_frames = niri_sequence('object')
    sky_frames = niri_sequence('sky2')

    adinputs = object_frames + sky_frames

    p = NIRIImage(adinputs)
    p.separateSky(frac_FOV=frac_FOV)

    # Check filenames just in case we ever change things such that astrodata
    # objects aren't held in memory anymore.
    out_obj_names = set(ad.phu['ORIGNAME'] for ad in p.streams['main'])
    out_sky_names = set(ad.phu['ORIGNAME'] for ad in p.streams['sky'])

    assert out_obj_names != out_sky_names

def test_separate_sky_cross_assign_frames(niri_sequence):

    niri_objects = niri_sequence('object')
    niri_skies = niri_sequence('sky1')

    obj_filenames = ','.join([ad.filename for ad in niri_objects])
    sky_filenames = ','.join([ad.filename for ad in niri_skies])

    adinputs = niri_objects
    adinputs.extend(niri_skies)

    p = NIRIImage(adinputs)
    p.separateSky(ref_obj=sky_filenames, ref_sky=obj_filenames)

    obj_names = ','.join([ad.phu['ORIGNAME'] for ad in p.streams['main']])
    sky_names = ','.join([ad.phu['ORIGNAME'] for ad in p.streams['sky']])

    assert obj_filenames == sky_names
    assert sky_filenames == obj_names

@pytest.mark.parametrize('frames', [0, -1])
def test_separate_sky_cross_assign_single_frames(frames, niri_sequence):
    """Test user assigning frames as sky or object."""

    niri_objects = niri_sequence('object')
    niri_skies = niri_sequence('sky1')

    manual_obj = niri_skies[frames].filename
    manual_sky = niri_objects[frames].filename

    obj_filenames = ','.join([ad.filename for ad in niri_objects])
    sky_filenames = ','.join([ad.filename for ad in niri_skies])

    adinputs = niri_objects
    adinputs.extend(niri_skies)

    p = NIRIImage(adinputs)
    p.separateSky(ref_obj=manual_obj,
                  ref_sky=manual_sky)

    obj_names = ','.join([ad.phu['ORIGNAME'] for ad in p.streams['main']])
    sky_names = ','.join([ad.phu['ORIGNAME'] for ad in p.streams['sky']])

    assert obj_filenames == sky_names
    assert sky_filenames == obj_names

@pytest.mark.parametrize('marker', ('object', 'sky'))
def test_separate_sky_assign_one_group(marker, niri_sequence):

    niri_objects = niri_sequence('object')
    niri_skies = niri_sequence('sky1')

    adinputs = niri_objects + niri_skies

    for ad in adinputs:
        for keyword in ('PWFS1_ST', 'PWFS2_ST', 'OIWFS_ST'):
            try:
                del ad.phu[keyword]
            except KeyError:
                pass
        ad.phu["OIWFS_ST"] = True

    filenames = {}
    filenames['object'] = ','.join([ad.filename for ad in niri_objects])
    filenames['sky'] = ','.join([ad.filename for ad in niri_skies])

    p = NIRIImage(adinputs)

    if marker == 'object':
        p.separateSky(ref_obj=filenames['object'])
    elif marker == 'sky':
        p.separateSky(ref_sky=filenames['sky'])

    obj_names = ','.join([ad.phu['ORIGNAME'] for ad in p.streams['main']])
    sky_names = ','.join([ad.phu['ORIGNAME'] for ad in p.streams['sky']])

    assert obj_names == filenames['object']
    assert sky_names == filenames['sky']

def test_separate_sky_assigned_header_keywords(niri_sequence):

    obj_frames = niri_sequence('object')
    sky_frames = niri_sequence('sky1')

    for ad in obj_frames:
        ad.phu['OBJFRAME'] = 'True'
    for ad in sky_frames:
        ad.phu['SKYFRAME'] = 'True'

    obj = obj_frames[0].filename
    sky = sky_frames[0].filename

    p = NIRIImage(obj_frames + sky_frames)
    p.separateSky(ref_obj=sky, ref_sky=obj)

    assert len(p.streams['main']) == len(p.streams['sky'])
    assert len(p.streams['main']) == len(obj_frames) + len(sky_frames)

@pytest.mark.parametrize('marker', ['object', 'sky1'])
def test_separate_sky_missing(marker, niri_sequence):

    input_length = len(niri_sequence(marker))
    p = NIRIImage(niri_sequence(marker))
    # Pass a non-existent filename to check handling of bad input
    p.separateSky(ref_sky='S20110101S0001.fits',
                  ref_obj='S20110101S0002.fits')

    assert len(p.streams['main']) == len(p.streams['sky'])
    assert len(p.streams['main']) == input_length

@pytest.mark.parametrize('groups', [('object', 'sky1'),
                                    ('object', 'sky1', 'sky2', 'sky3')])
def test_separate_sky_proximity(groups, niri_sequence):

    adinputs = []
    for marker in groups:
        adinputs.extend(niri_sequence(marker))

    # Set all inputs to be "guided" data to remove that as a means of
    # determining skies to force the use of proximity of groups.
    for ad in adinputs:
        for keyword in ('PWFS1_ST', 'PWFS2_ST', 'OIWFS_ST'):
            try:
                del ad.phu[keyword]
            except KeyError:
                pass
        ad.phu["OIWFS_ST"] = True

    p = NIRIImage(adinputs)
    p.separateSky()

    # In both cases the proximity assignment should put half the groups as
    # 'object' and half as 'sky'.
    assert len(p.streams['main']) * 2 == len(adinputs)
    assert len(p.streams['sky']) * 2 == len(adinputs)

# @pytest.mark.parametrize('frame', ['object', 'sky'])#, 'mixed'])
# def test_list_ra_dec(frame, niri_sequence):

#     adinputs = niri_sequence(frame)
#     for ad in adinputs:
#         # print(f'{ad.filename}')
#         print(ad.phu['RA'], ad.phu['RAOFFSET'], sep=', ')
#     for ad in adinputs:
#         print(ad.phu['DEC'], ad.phu['DECOFFSE'], sep=',')

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
