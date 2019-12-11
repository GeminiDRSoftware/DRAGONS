from datetime import datetime, timedelta

import astrodata
from astrodata import fits, Provenance, ProvenanceHistory
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
    provenance_added_by = "provenance_added_by"

    ad.add_provenance(Provenance(timestamp, filename, md5, provenance_added_by))

    provenance = ad.get_provenance()

    assert(len(provenance) == 1)

    p = provenance[0]

    assert(p.timestamp == timestamp)
    assert(p.filename == filename)
    assert(p.md5 == md5)
    assert(p.provenance_added_by == provenance_added_by)


def tests_add_get_provenance_history():
    ad = _dummy_astrodata_fits()
    timestamp_start = datetime.now()
    timestamp_end = timestamp_start + timedelta(days=1)
    primitive = "primitive"
    args = "args"

    ad.add_provenance_history(ProvenanceHistory(timestamp_start, timestamp_end, primitive, args))

    provenance_history = ad.get_provenance_history()

    assert(len(provenance_history) == 1)

    ph = provenance_history[0]

    assert(ph.timestamp_start == timestamp_start)
    assert(ph.timestamp_stop == timestamp_end)
    assert(ph.primitive == primitive)
    assert(ph.args == args)
