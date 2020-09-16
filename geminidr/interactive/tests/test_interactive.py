from astropy.units import Quantity
from numpy.core.multiarray import ndarray

from geminidr.interactive.deprecated.deprecated_interactive import _dequantity


def test_dequantity():
    x = ndarray([0,1,2])
    y = ndarray([0,1,2])

    x2, y2 = _dequantity(x, y)

    assert((x == x2).all())
    assert((y == y2).all())


def test_dequantity_quantities():
    xnd = ndarray([0,1,2])
    ynd = ndarray([1,2,3])

    x = Quantity(xnd)
    y = Quantity(ynd)

    x2, y2 = _dequantity(x, y)

    assert((xnd == x2).all())
    assert((ynd == y2).all())
