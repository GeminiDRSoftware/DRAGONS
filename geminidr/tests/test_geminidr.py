import pytest

import geminidr
from astrodata.testing import download_from_archive
from geminidr.gmos.primitives_gmos_longslit import GMOSLongslit
from recipe_system.reduction.coreReduce import UnrecognizedParameterException

import astrodata
import gemini_instruments  # needed to initialize instrument support


@pytest.mark.parametrize('parm, expected',
                         [
                             ([('nteractive', True)], 'Parameter nteractive not recognized'),
                             ([('calculateSensitivity:nteractive', True)], 'Parameter nteractive not recognized'),
                             ([('alculateSensitivity:interactive', True)], 'Primitive alculateSensitivity not recognized'),
                             ([('calculateSensitivity:Interactive', True)],
                              'Parameter Interactive not recognized, did you mean interactive?'),
                             ([('CalculateSensitivity:interactive', True)],
                              'Primitive CalculateSensitivity not recognized, did you mean calculateSensitivity?'),
                             ([('calculateSensitivity:interactive:pickles', True)],
                              'Expecting parameter or primitive:parameter in -p user parameters')
                         ])
def test_unrecognized_uparm(parm, expected):
    """ Test handling of unrecognized user parameters. """
    # quick argparse to mimic a call from reduce.py
    testfile = download_from_archive("N20160524S0119.fits")

    with pytest.raises(UnrecognizedParameterException) as upe:
        GMOSLongslit(astrodata.open(testfile), mode='sq', ucals={}, uparms=parm, upload=None, config_file=None)
    assert expected in str(upe.value)


def test_parm_recheck_bad_values_every_time():
    """ Test handling of unrecognized user parameters. """
    # quick argparse to mimic a call from reduce.py
    testfile = download_from_archive("N20160524S0119.fits")

    with pytest.raises(UnrecognizedParameterException) as upe:
        GMOSLongslit(astrodata.open(testfile), mode='sq', ucals={}, uparms=[('nteractive', True)], upload=None,
                     config_file=None)
    with pytest.raises(UnrecognizedParameterException) as upe2:
        GMOSLongslit(astrodata.open(testfile), mode='sq', ucals={}, uparms=[('nteractive', True)], upload=None,
                     config_file=None)
    assert(upe2 is not None)  # subsequent runs with bad parameters should still abort


def test_parm_dont_recheck_same_passing_parms(monkeypatch):
    """ Test handling of unrecognized user parameters. """
    # quick argparse to mimic a call from reduce.py
    testfile = download_from_archive("N20160524S0119.fits")

    GMOSLongslit(astrodata.open(testfile), mode='sq', ucals={}, uparms=[('interactive', True)], upload=None,
                 config_file=None)

    def mock_find_similar_names(name, valid_names):
        # Even though we monkeypatch this in, this function will not be called.
        # The PrimitivesBase will see that we are passing the same uparms and will
        # skip the parameter validation step, where this would normally be called
        return ["this_causes_check_to_throw_exception",]
    monkeypatch.setattr(geminidr, "_find_similar_names", mock_find_similar_names)
    GMOSLongslit(astrodata.open(testfile), mode='sq', ucals={}, uparms=[('interactive', True)], upload=None,
                 config_file=None)
