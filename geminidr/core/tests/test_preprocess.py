# pytest suite
"""
Tests for primitives_preprocess.

This is a suite of tests to be run with pytest.

To run:
    1) Set the environment variable GEMPYTHON_TESTDATA to the path that
       contains the directories with the test data.
       Eg. /net/chara/data2/pub/gempython_testdata/
    2) From the ??? (location): pytest -v --capture=no
"""

# TODO @bquint: clean up these tests

from itertools import count
import os
import pytest

from copy import deepcopy

import astrodata
import gemini_instruments

# from . import ad_compare
from geminidr.niri.primitives_niri_image import NIRIImage
from geminidr.gmos.primitives_gmos_image import GMOSImage
from gempy.utils import logutils

TESTDATAPATH = os.getenv('GEMPYTHON_TESTDATA', '.')
logfilename = 'test_preprocess.log'


@pytest.fixture
def niri_image():
    """Create a fake NIRI image.

    Optional
    --------
    keywords : dict
        A dictionary with keys equal to FITS header keywords, whose values
        will be propogated to the new image.

    """
    import astrofaker
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

@pytest.mark.parametrize('use_all',
                         [False, True])
def test_associate_sky_use_all(use_all, niri_sequence):

    objects = niri_sequence('object')
    skies1 = niri_sequence('sky1')
    skies2 = niri_sequence('sky2')
    skies3 = niri_sequence('sky3')

    p = NIRIImage(objects + skies1 + skies2 + skies3)
    p.separateSky()
    # This test checks that minimum distance is respected, unless
    # 'use_all' == True.
    p.associateSky(distance=320, use_all=use_all)

    for ad in p.showList():
        skies = set([row[0].replace('_skyAssociated', '')
                     for row in ad.SKYTABLE])
        assert (skies1[0].phu['ORIGNAME'] in skies) == use_all

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
