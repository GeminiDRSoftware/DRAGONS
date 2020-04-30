import pytest
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.errors import RecipeNotFound


def test_primitive_not_found(cache_file_from_archive):
    testfile = cache_file_from_archive("N20160524S0119.fits")

    red = Reduce()
    red.files = [testfile]
    red.recipename = 'foobar'
    with pytest.raises(RecipeNotFound, match='No primitive named foobar'):
        red.runr()


def test_mode_not_found(cache_file_from_archive):
    testfile = cache_file_from_archive("N20160524S0119.fits")

    red = Reduce()
    red.files = [testfile]
    red.mode = 'aa'
    with pytest.raises(RecipeNotFound,
                       match="GMOS recipes do not define a 'aa' recipe"):
        red.runr()
