from datetime import datetime, timedelta, timezone
import os

import numpy as np
import pytest

import astrodata
from astrodata import fits
from astrodata.testing import download_from_archive
from astrodata.provenance import (add_provenance, add_history, clone_provenance,
                                  clone_history)


@pytest.fixture
def ad():
    phu = fits.PrimaryHDU()
    hdu = fits.ImageHDU(data=np.ones((10, 10)), name='SCI')
    return astrodata.create(phu, [hdu])


@pytest.fixture
def ad2():
    phu = fits.PrimaryHDU()
    hdu = fits.ImageHDU(data=np.ones((10, 10)), name='SCI')
    return astrodata.create(phu, [hdu])


def test_add_get_provenance(ad):
    timestamp = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    filename = "filename"
    md5 = "md5"
    primitive = "provenance_added_by"

    # if md5 is None, provenance is added with empty string as md5
    add_provenance(ad, filename, None, primitive)
    assert len(ad.PROVENANCE) == 1
    assert tuple(ad.PROVENANCE[0])[1:] == (filename, '', primitive)

    add_provenance(ad, filename, md5, primitive, timestamp=timestamp)
    assert len(ad.PROVENANCE) == 2
    assert tuple(ad.PROVENANCE[1]) == (timestamp, filename, md5, primitive)

    # entry is updated and a default timestamp is created
    add_provenance(ad, filename, md5, primitive)
    assert len(ad.PROVENANCE) == 2
    assert tuple(ad.PROVENANCE[1])[1:] == (filename, md5, primitive)

    # add new entry
    add_provenance(ad, filename, 'md6', 'other primitive')
    assert len(ad.PROVENANCE) == 3
    assert tuple(ad.PROVENANCE[1])[1:] == (filename, md5, primitive)
    assert tuple(ad.PROVENANCE[2])[1:] == (filename, 'md6', 'other primitive')


def test_add_duplicate_provenance(ad):
    timestamp = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    filename = "filename"
    md5 = "md5"
    primitive = "provenance_added_by"

    add_provenance(ad, filename, md5, primitive, timestamp=timestamp)
    add_provenance(ad, filename, md5, primitive, timestamp=timestamp)

    # was a dupe, so should have been skipped
    assert len(ad.PROVENANCE) == 1


def test_add_get_history(ad):
    timestamp_start = datetime.now(timezone.utc).replace(tzinfo=None)
    timestamp_end = (timestamp_start +
                     timedelta(days=1)).isoformat()
    timestamp_start = timestamp_start.isoformat()
    primitive = "primitive"
    args = "args"

    add_history(ad, timestamp_start, timestamp_end, primitive, args)
    assert len(ad.HISTORY) == 1
    assert tuple(ad.HISTORY[0]) == (primitive, args, timestamp_start,
                                    timestamp_end)

    add_history(ad, timestamp_start, timestamp_end, 'another primitive', args)
    assert len(ad.HISTORY) == 2
    assert tuple(ad.HISTORY[0]) == (primitive, args, timestamp_start,
                                    timestamp_end)
    assert tuple(ad.HISTORY[1]) == ('another primitive', args,
                                    timestamp_start, timestamp_end)


def test_add_dupe_history(ad):
    timestamp_start = datetime.now(timezone.utc).replace(tzinfo=None)
    timestamp_end = (timestamp_start + timedelta(days=1)).isoformat()
    timestamp_start = timestamp_start.isoformat()
    primitive = "primitive"
    args = "args"

    add_history(ad, timestamp_start, timestamp_end, primitive, args)
    add_history(ad, timestamp_start, timestamp_end, primitive, args)

    # was a dupe, should have skipped 2nd add
    assert len(ad.HISTORY) == 1


def test_clone_provenance(ad, ad2):
    timestamp = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    filename = "filename"
    md5 = "md5"
    primitive = "provenance_added_by"

    add_provenance(ad, filename, md5, primitive, timestamp=timestamp)

    clone_provenance(ad.PROVENANCE, ad2)

    assert len(ad2.PROVENANCE) == 1
    assert tuple(ad2.PROVENANCE[0]) == (timestamp, filename, md5, primitive)


def test_clone_history(ad, ad2):
    timestamp_start = datetime.now(timezone.utc).replace(tzinfo=None)
    timestamp_end = (timestamp_start + timedelta(days=1)).isoformat()
    timestamp_start = timestamp_start.isoformat()
    primitive = "primitive"
    args = "args"

    add_history(ad, timestamp_start, timestamp_end, primitive, args)

    clone_history(ad.HISTORY, ad2)

    assert len(ad2.HISTORY) == 1
    assert tuple(ad2.HISTORY[0]) == (primitive, args, timestamp_start,
                                     timestamp_end)


@pytest.fixture(scope='module')
def BPM_PROVHISTORY():
    """
    BPM file with PROVHISTORY (old name for HISTORY)
    """
    return download_from_archive("bpm_20220128_gmos-s_Ham_11_full_12amp.fits")


@pytest.mark.dragons_remote_data
def test_convert_provhistory(tmpdir, BPM_PROVHISTORY):
    ad = astrodata.open(BPM_PROVHISTORY)

    # This file (should) use the old PROVHISTORY extname
    assert hasattr(ad, 'PROVHISTORY')

    # When we add history, that should get converted to HISTORY
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    add_history(ad, now, now, "primitive", "args")
    assert not hasattr(ad, 'PROVHISTORY')
    assert hasattr(ad, 'HISTORY')

    # and if we write the file, it should have a HISTORY extname
    # and not a PROVHISTORY extname
    testfile = os.path.join(str(tmpdir), 'temp.fits')
    ad.path = testfile
    ad.write()
    assert os.path.exists(testfile)

    ad2 = astrodata.open(testfile)
    assert hasattr(ad2, 'HISTORY')
    assert not hasattr(ad2, 'PROVHISTORY')

    # and should have that new history record we added
    hist = ad.HISTORY[-1]
    assert hist['timestamp_start'] == now
    assert hist['timestamp_stop'] == now
    assert hist['primitive'] == 'primitive'
    assert hist['args'] == 'args'

    os.remove(testfile)
