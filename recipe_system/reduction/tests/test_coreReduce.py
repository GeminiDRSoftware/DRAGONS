import pytest
from astrodata.testing import download_from_archive
from recipe_system.reduction.coreReduce import Reduce, UnrecognizedParameterException
from recipe_system.utils.errors import RecipeNotFound
from recipe_system.utils.reduce_utils import buildParser, normalize_args, normalize_upload
from recipe_system import __version__ as rs_version


@pytest.mark.dragons_remote_data
def test_primitive_not_found():
    testfile = download_from_archive("N20160524S0119.fits")

    red = Reduce()
    red.files = [testfile]
    red.recipename = 'foobar'
    with pytest.raises(RecipeNotFound, match='No primitive named foobar'):
        red.runr()


@pytest.mark.dragons_remote_data
def test_mode_not_found():
    testfile = download_from_archive("N20160524S0119.fits")

    red = Reduce()
    red.files = [testfile]
    red.mode = 'aa'
    with pytest.raises(RecipeNotFound,
                       match="GMOS recipes do not define a 'aa' recipe"):
        red.runr()


@pytest.mark.parametrize('parm, expected',
                         [
                             ('nteractive=True', 'Parameter nteractive not recognized'),
                             ('calculateSensitivity:nteractive=True', 'Parameter nteractive not recognized'),
                             ('alculateSensitivity:interactive=True', 'Primitive alculateSensitivity not found'),
                             ('calculateSensitivity:Interactive=True',
                              'Parameter Interactive not found, did you mean interactive?'),
                             ('CalculateSensitivity:interactive=True',
                              'Primitive CalculateSensitivity not found, did you mean calculateSensitivity?'),
                             ('calculateSensitivity:interactive:pickles=True',
                              'Expecting parameter or primitive:parameter in -p user parameters')
                         ])
def test_unrecognized_uparm(parm, expected, monkeypatch):
    """ Test handling of unrecognized user parameters. """
    # quick argparse to mimic a call from reduce.py
    testfile = download_from_archive("N20160524S0119.fits")

    parser = buildParser(rs_version)
    args = parser.parse_args(['-p', parm, '-r', 'calculateSensitivity', "N20160524S0119.fits"])
    args = normalize_args(args)
    args.upload = normalize_upload(args.upload)

    red = Reduce(sys_args=args)
    red.files = [testfile]
    with pytest.raises(UnrecognizedParameterException) as upe:
        red.runr()
    assert expected in str(upe.value)
