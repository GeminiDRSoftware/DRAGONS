import pytest
from astrodata.testing import download_from_archive
from recipe_system.reduction.coreReduce import Reduce, UnrecognizedParameterException
from recipe_system.utils.errors import RecipeNotFound
from recipe_system.utils.reduce_utils import buildParser, normalize_args, normalize_upload


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


@pytest.mark.parameterize('parm, expected',
                          [
                              ('calculateSensitivity:nteractive=True', 'Parameter nteractive not found'),
                              ('alculateSensitivity:interactive=True', 'Primitive alculateSensitivity not found'),
                              ('calculateSensitivity:Interactive=True',
                               'Parameter Interactive not found, did you mean interactive?'),
                              ('CalculateSensitivity:interactive=True',
                               'Primitive CalculateSensitivity not found, did you mean calculateSensitivity?'),
                              ('calculateSensitivity:interactive:pickles=True',
                               'Expecting parameter or primitive:parameter in -p user parameters')
                          ])
def test_unrecognized_uparm(parm, expected):
    """ Test handling of unrecognized user parameters. """
    # quick argparse to mimic a call from reduce.py
    parser = buildParser()
    args = parser.parse_args(['-p', parm, '-r', 'calculateSensitivity', 'calculatesensitivity_example.fits'])
    args = normalize_args(args)
    args.upload = normalize_upload(args.upload)

    red = Reduce(sys_args=args)

    with pytest.raises(UnrecognizedParameterException) as upe:
        red.runr()
    assert expected in str(upe.value)
