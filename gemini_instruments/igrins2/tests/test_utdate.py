from pathlib import Path
from astropy.io import fits
import datetime

import astrodata
import igrins_instruments

from . import test_data
from importlib.resources import files

def test_utdatetime():
    dataroot = files(test_data)
    datadir = dataroot / "sample_sky"
    fn = datadir / "N20240429S0204_H.fits"
    ad = astrodata.open(fn)
    dt = datetime.datetime.fromisoformat(ad[0].hdr["UTSTART"])
    assert ad[0].ut_datetime() == dt

    hdr = ad[0].hdr
    hdr["UTDATETI"] = hdr["UTSTART"]
    del hdr["UTSTART"]
    assert ad[0].ut_datetime() == dt

    assert ad.ut_datetime() == dt
