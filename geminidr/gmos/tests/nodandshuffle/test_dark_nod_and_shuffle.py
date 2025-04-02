import pytest

import os, re

import astrodata, gemini_instruments
from geminidr.gmos.primitives_gmos_longslit import GMOSLongslit

datasets = [('N20190926S0130_prepared.fits', ''),
            ('N20180909S0124_ADUToElectrons.fits', 'N20180908S0104_dark.fits')]

def fake_dark(adinputs, result):
    class fakeCalReturn:
        def __init__(self, filename):
            if filename == '':
                self.files = [None]
            else:
                self.files = [filename]

    fclass = fakeCalReturn(result)
    return fclass

@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize('ad, result', datasets, indirect=["ad"])
def test_no_dark_hamamatsu(ad, result, monkeypatch, path_to_inputs):
    p = GMOSLongslit([ad])
    pathresult = result
    if pathresult != '':
        pathresult = os.path.join(path_to_inputs, result)

    monkeypatch.setattr(p.caldb, 'get_processed_dark', lambda x: fake_dark(x, pathresult))
    p.darkCorrect()
    dark_used = ad.phu.get('DARKIM', '')
    assert dark_used == result


@pytest.fixture(scope='function')
def ad(path_to_inputs, request):
    """Return AD object in input directory"""
    path = os.path.join(path_to_inputs, request.param)
    if os.path.exists(path):
        return astrodata.from_file(path)
    raise FileNotFoundError(path)
