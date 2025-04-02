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
                              'Expecting parameter or primitive:parameter in -p user parameters'),
                             ([('calculateSensitivity:intiractive', True)],
                              'Parameter intiractive not recognized, did you mean interactive?')
                         ])
def test_unrecognized_uparm(parm, expected):
    """ Test handling of unrecognized user parameters. """
    # quick argparse to mimic a call from reduce.py
    testfile = download_from_archive("N20160524S0119.fits")

    with pytest.raises(UnrecognizedParameterException) as upe:
        GMOSLongslit(astrodata.from_file(testfile), mode='sq', ucals={}, uparms=parm, upload=None, config_file=None)
    assert expected in str(upe.value)


if __name__ == "__main__":
    pytest.main()
