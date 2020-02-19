from datetime import datetime, timedelta

import astrodata
from astrodata import fits
import numpy as np

from astrodata.provenance import PROVENANCE_DATE_FORMAT, add_provenance, add_provenance_history


def _dummy_astrodata_fits():
    data_array = np.ones((100, 100))

    phu = fits.PrimaryHDU()
    hdu = fits.ImageHDU(data=data_array, name='SCI')

    ad = astrodata.create(phu, [hdu])

    return ad


def test_add_get_provenance():
    ad = _dummy_astrodata_fits()
    timestamp = datetime.now().strftime(PROVENANCE_DATE_FORMAT)
    filename = "filename"
    md5 = "md5"
    provenance_added_by = "provenance_added_by"

    add_provenance(ad, filename, md5, provenance_added_by, timestamp=timestamp)

    provenance = ad.PROVENANCE

    assert(len(provenance) == 1)

    p = provenance[0]

    assert(p[0] == timestamp)
    assert(p[1] == filename)
    assert(p[2] == md5)
    assert(p[3] == provenance_added_by)


def test_add_duplicate_provenance():
    ad = _dummy_astrodata_fits()
    timestamp = datetime.now().strftime(PROVENANCE_DATE_FORMAT)
    filename = "filename"
    md5 = "md5"
    provenance_added_by = "provenance_added_by"

    add_provenance(ad, filename, md5, provenance_added_by, timestamp=timestamp)
    add_provenance(ad, filename, md5, provenance_added_by, timestamp=timestamp)

    # was a dupe, so should have been skipped
    assert(len(ad.PROVENANCE) == 1)


def test_add_get_provenance_history():
    ad = _dummy_astrodata_fits()
    timestamp_start = datetime.now()
    timestamp_end = (timestamp_start + timedelta(days=1)).strftime(PROVENANCE_DATE_FORMAT)
    timestamp_start = timestamp_start.strftime(PROVENANCE_DATE_FORMAT)
    primitive = "primitive"
    args = "args"

    add_provenance_history(ad, timestamp_start, timestamp_end, primitive, args)

    provenance_history = ad.PROVENANCE_HISTORY

    assert(len(provenance_history) == 1)

    ph = provenance_history[0]

    assert(ph[0] == timestamp_start)
    assert(ph[1] == timestamp_end)
    assert(ph[2] == primitive)
    assert(ph[3] == args)


def test_add_dupe_provenance_history():
    ad = _dummy_astrodata_fits()
    timestamp_start = datetime.now()
    timestamp_end = (timestamp_start + timedelta(days=1)).strftime(PROVENANCE_DATE_FORMAT)
    timestamp_start = timestamp_start.strftime(PROVENANCE_DATE_FORMAT)
    primitive = "primitive"
    args = "args"

    add_provenance_history(ad, timestamp_start, timestamp_end, primitive, args)
    add_provenance_history(ad, timestamp_start, timestamp_end, primitive, args)

    # was a dupe, should have skipped 2nd add
    assert(len(ad.PROVENANCE_HISTORY) == 1)
