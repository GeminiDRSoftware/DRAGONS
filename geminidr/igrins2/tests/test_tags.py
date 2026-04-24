import os

import pytest

import astrodata, gemini_instruments
from astrodata.testing import download_from_archive


TAGS_FIXTURE_DATA = {('IGRINS-2', 'N20260303S0289.fits'):  # arc
                         ['IGRINS-2', 'IGRINS', 'GEMINI', 'NORTH', 'RAW', 'UNPREPARED', 'BUNDLE', 'LAMPOFF'],
                     ('IGRINS-2', 'N20260303S0290.fits'):  # flat
                         ['IGRINS-2', 'IGRINS', 'GEMINI', 'NORTH', 'RAW', 'UNPREPARED', 'BUNDLE', 'LAMPOFF', 'CAL', 'FLAT', 'GCALFLAT'],
                     ('IGRINS-2', 'N20260228S0506.fits'):  # dark (no DARK?)
                         ['IGRINS-2', 'IGRINS', 'GEMINI', 'NORTH', 'RAW', 'UNPREPARED', 'BUNDLE', 'LAMPOFF'],
                     ('IGRINS-2', 'N20260228S0200.fits'):  # telluric standard
                         ['IGRINS-2', 'IGRINS', 'GEMINI', 'NORTH', 'RAW', 'UNPREPARED', 'BUNDLE', 'LAMPOFF', 'CAL', 'SIDEREAL', 'STANDARD'],
                     ('IGRINS-2', 'N20260303S0044.fits'):  # blank sky
                         ['IGRINS-2', 'IGRINS', 'GEMINI', 'NORTH', 'RAW', 'UNPREPARED', 'BUNDLE', 'LAMPOFF', 'CAL', 'SIDEREAL', 'SKY']
                     }


@pytest.mark.parametrize("instr,filename,tag_set",
                         ([*k]+[v] for k, v in TAGS_FIXTURE_DATA.items()))
def test_tags(instr, filename, tag_set):
    ad = astrodata.open(download_from_archive(filename))
    assert ad.tags == set(tag_set)
