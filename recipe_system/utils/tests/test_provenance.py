from datetime import datetime, timedelta

import astrodata
from astrodata import fits
from recipe_system.utils.provenance import add_provenance, get_provenance, add_provenance_history, \
    get_provenance_history
import numpy as np


def _dummy_astrodata_fits():
    data_array = np.ones((100, 100))

    phu = fits.PrimaryHDU()
    hdu = fits.ImageHDU(data=data_array, name='SCI')

    ad = astrodata.create(phu, [hdu])

    return ad


def tests_add_get_provenance():
    ad = _dummy_astrodata_fits()
    timestamp = datetime.now()
    filename = "filename"
    md5 = "md5"
    primitive = "primitive"

    add_provenance(ad, timestamp, filename, md5, primitive)

    provenance = get_provenance(ad)

    assert(len(provenance) == 1)

    row = provenance[0]

    assert(row["timestamp"] == timestamp)
    assert(row["filename"] == filename)
    assert(row["md5"] == md5)
    assert(row["primitive"] == primitive)


def tests_add_get_provenance_history():
    ad = _dummy_astrodata_fits()
    timestamp_start = datetime.now()
    timestamp_end = timestamp_start + timedelta(days=1)
    primitive = "primitive"
    args = "args"

    add_provenance_history(ad, timestamp_start, timestamp_end, primitive, args)

    provenance_history = get_provenance_history(ad)

    assert(len(provenance_history) == 1)

    row = provenance_history[0]

    assert(row["timestamp_start"] == timestamp_start)
    assert(row["timestamp_end"] == timestamp_end)
    assert(row["primitive"] == primitive)
    assert(row["args"] == args)
