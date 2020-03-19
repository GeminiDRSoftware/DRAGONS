import numpy as np
import pytest
import astropy.units as u
from astropy.table import Table
from astrodata.fits import header_for_table


def test_header_for_table():
    tbl = Table([[[1], [2, 3], [3, 4, 5]]])
    with pytest.raises(TypeError,
                       match=r"Variable length arrays .* are not supported"):
        header_for_table(tbl)

    tbl = Table([np.arange(2 * 3 * 4).reshape(3, 2, 4),
                 [1.0, 2.0, 3.0],
                 ['aa', 'bb', 'cc'],
                 [[True, False], [True, False], [True, False]]],
                names='abcd')
    tbl['b'].unit = u.arcsec
    hdr = header_for_table(tbl)
    assert hdr['TFORM1'] == '8K'
    assert hdr['TDIM1'] == 2
    assert hdr['TFORM4'] == '2X'
    assert hdr['TUNIT2'] == 'arcsec'
