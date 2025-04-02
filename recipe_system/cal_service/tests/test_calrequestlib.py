# Test calrequestlib and in particular the GHOST bundle dictionary descriptors being handled
import pytest
import io
import os
from glob import glob

import astrodata, gemini_instruments
import astrodata.testing

from recipe_system import cal_service
from recipe_system.cal_service.calrequestlib import get_cal_requests
from recipe_system.config import globalConf

from geminidr.gmos.primitives_gmos_longslit import GMOSClassicLongslit


@pytest.mark.dragons_remote_data
def test_get_cal_requests_dictdescriptor():
    path = astrodata.testing.download_from_archive("S20221209S0007.fits")
    ad = astrodata.from_file(path)
    requests = get_cal_requests([ad], 'bias')
    # This descriptor works off the decomposed per-arm values for x/y binning
    # so it's a good test that this worked with a dictionary-based descriptor
    assert requests[0].descriptors['detector_binning'] == '1x1'
