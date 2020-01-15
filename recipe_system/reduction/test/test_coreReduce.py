import pytest
from astrodata.testing import download_from_archive
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.errors import RecipeNotFound


@pytest.mark.dragons_remote_data
def test_primitive_not_found():
    testfile = download_from_archive("N20160524S0119.fits", path="GMOS")

    red = Reduce()
    red.files = [testfile]
    red.recipename = 'foobar'
    with pytest.raises(RecipeNotFound, match='No primitive named foobar'):
        red.runr()


@pytest.mark.dragons_remote_data
def test_mode_not_found():
    testfile = download_from_archive("N20160524S0119.fits", path="GMOS")

    red = Reduce()
    red.files = [testfile]
    red.mode = 'aa'
    with pytest.raises(RecipeNotFound,
                       match="GMOS recipes do not define a 'aa' recipe"):
        red.runr()
