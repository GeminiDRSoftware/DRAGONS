from datetime import datetime, timedelta

import numpy as np
import pytest

import astrodata
from astrodata import fits
from astrodata.provenance import (PROVENANCE_DATE_FORMAT, add_provenance,
                                  add_provenance_history, clone_provenance,
                                  clone_provenance_history)


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
    timestamp = datetime.now().strftime(PROVENANCE_DATE_FORMAT)
    filename = "filename"
    md5 = "md5"
    primitive = "provenance_added_by"

    # if md5 is None, nothing is added
    add_provenance(ad, filename, None, primitive)
    assert not hasattr(ad, 'PROVENANCE')

    add_provenance(ad, filename, md5, primitive, timestamp=timestamp)
    assert len(ad.PROVENANCE) == 1
    assert tuple(ad.PROVENANCE[0]) == (timestamp, filename, md5, primitive)

    # entry is updated and a default timestamp is created
    add_provenance(ad, filename, md5, primitive)
    assert len(ad.PROVENANCE) == 1
    assert tuple(ad.PROVENANCE[0])[1:] == (filename, md5, primitive)

    # add new entry
    add_provenance(ad, filename, 'md6', 'other primitive')
    assert len(ad.PROVENANCE) == 2
    assert tuple(ad.PROVENANCE[0])[1:] == (filename, md5, primitive)
    assert tuple(ad.PROVENANCE[1])[1:] == (filename, 'md6', 'other primitive')


def test_add_duplicate_provenance(ad):
    timestamp = datetime.now().strftime(PROVENANCE_DATE_FORMAT)
    filename = "filename"
    md5 = "md5"
    primitive = "provenance_added_by"

    add_provenance(ad, filename, md5, primitive, timestamp=timestamp)
    add_provenance(ad, filename, md5, primitive, timestamp=timestamp)

    # was a dupe, so should have been skipped
    assert len(ad.PROVENANCE) == 1


def test_add_get_provenance_history(ad):
    timestamp_start = datetime.now()
    timestamp_end = (timestamp_start +
                     timedelta(days=1)).strftime(PROVENANCE_DATE_FORMAT)
    timestamp_start = timestamp_start.strftime(PROVENANCE_DATE_FORMAT)
    primitive = "primitive"
    args = "args"

    add_provenance_history(ad, timestamp_start, timestamp_end, primitive, args)
    assert len(ad.PROVENANCE_HISTORY) == 1
    assert tuple(ad.PROVENANCE_HISTORY[0]) == (timestamp_start, timestamp_end,
                                               primitive, args)

    add_provenance_history(ad, timestamp_start, timestamp_end,
                           'another primitive', args)
    assert len(ad.PROVENANCE_HISTORY) == 2
    assert tuple(ad.PROVENANCE_HISTORY[0]) == (timestamp_start, timestamp_end,
                                               primitive, args)
    assert tuple(ad.PROVENANCE_HISTORY[1]) == (timestamp_start, timestamp_end,
                                               'another primitive', args)


def test_add_dupe_provenance_history(ad):
    timestamp_start = datetime.now()
    timestamp_end = (timestamp_start +
                     timedelta(days=1)).strftime(PROVENANCE_DATE_FORMAT)
    timestamp_start = timestamp_start.strftime(PROVENANCE_DATE_FORMAT)
    primitive = "primitive"
    args = "args"

    add_provenance_history(ad, timestamp_start, timestamp_end, primitive, args)
    add_provenance_history(ad, timestamp_start, timestamp_end, primitive, args)

    # was a dupe, should have skipped 2nd add
    assert len(ad.PROVENANCE_HISTORY) == 1


def test_clone_provenance(ad, ad2):
    timestamp = datetime.now().strftime(PROVENANCE_DATE_FORMAT)
    filename = "filename"
    md5 = "md5"
    primitive = "provenance_added_by"

    add_provenance(ad, filename, md5, primitive, timestamp=timestamp)

    clone_provenance(ad.PROVENANCE, ad2)

    assert len(ad2.PROVENANCE) == 1
    assert tuple(ad2.PROVENANCE[0]) == (timestamp, filename, md5, primitive)


def test_clone_provenance_history(ad, ad2):
    timestamp_start = datetime.now()
    timestamp_end = (timestamp_start +
                     timedelta(days=1)).strftime(PROVENANCE_DATE_FORMAT)
    timestamp_start = timestamp_start.strftime(PROVENANCE_DATE_FORMAT)
    primitive = "primitive"
    args = "args"

    add_provenance_history(ad, timestamp_start, timestamp_end, primitive, args)

    clone_provenance_history(ad.PROVENANCE_HISTORY, ad2)

    assert len(ad2.PROVENANCE_HISTORY) == 1
    assert tuple(ad2.PROVENANCE_HISTORY[0]) == (timestamp_start, timestamp_end,
                                                primitive, args)
