#!/usr/bin/env python
import pytest

import astrodata
import gemini_instruments

from astrodata.testing import download_from_archive
from geminidr.f2.primitives_f2_image import F2Image

test_files = [
    "S20131121S0094.fits",
    "S20131126S1111.fits",
    "S20131126S1112.fits",
    "S20131126S1113.fits",
    "S20160112S0080.fits",
    "S20170103S0032.fits",
]


@pytest.fixture(scope='module')
def input_ad(request):
    filename = request.param
    path = download_from_archive(filename)
    ad = astrodata.from_file(path)
    return ad


@pytest.mark.f2
@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("input_ad", test_files, indirect=True)
def test_select_from_inputs(change_working_dir, input_ad):

    with change_working_dir():
        with open("recursion.log", 'w') as _log:

            p = F2Image([input_ad])
            p.prepare(bad_wcs="ignore")
            p.addVAR(read_noise=True)

            try:
                print('Running tests on file: {:s}'.format(input_ad.filename))
                p.selectFromInputs(tags="DARK", outstream="darks")
                p.showInputs(stream="darks")
                p.showInputs()

                print("{:15s} OK".format(input_ad.filename))
                _log.write("\n{:15s} OK".format(input_ad.filename))

            except RecursionError as re:
                print("{:15s} FAIL".format(input_ad.filename))
                _log.write("\n{:15s} FAIL".format(input_ad.filename))

            del input_ad
            del p


if __name__ == '__main__':
    pytest.main()
