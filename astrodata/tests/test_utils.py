import numpy as np
import pytest
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astrodata.fits import header_for_table, card_filter, update_header
from astrodata import Section


def test_header_for_table():
    tbl = Table([[[1], [2, 3], [3, 4, 5]]], dtype=['object'])
    with pytest.raises(ValueError, match=r"Illegal format `object`"):
        header_for_table(tbl)

    tbl = Table([np.arange(2 * 3 * 4).reshape(3, 2, 4),
                 [1.0, 2.0, 3.0],
                 ['aa', 'bb', 'cc'],
                 [[True, False], [True, False], [True, False]]],
                names='abcd')
    tbl['b'].unit = u.arcsec
    hdr = header_for_table(tbl)
    assert hdr['TFORM1'] == '8K'
    assert hdr['TDIM1'] == '(4,2)'
    assert hdr['TFORM4'] == '2L'
    assert hdr['TUNIT2'] == 'arcsec'


def test_card_filter():
    hdr = fits.Header(dict(zip('ABCDE', range(5))))
    assert [c.keyword for c in card_filter(hdr.cards, include='ABC')] == \
        ['A', 'B', 'C']
    assert [c.keyword for c in card_filter(hdr.cards, exclude='AB')] == \
        ['C', 'D', 'E']


def test_update_header():
    hdra = fits.Header({'INSTRUME': 'darkimager', 'OBJECT': 'M42'})
    hdra.add_comment('A super useful comment')
    hdra.add_history('This is historic')
    assert update_header(hdra, hdra) is hdra

    hdrb = fits.Header({'OBJECT': 'IO', 'EXPTIME': 42})
    hdrb.add_comment('A super useful comment')
    hdrb.add_comment('Another comment')
    hdrb.add_history('This is historic')
    hdrb.add_history('And not so useful')

    hdr = update_header(hdra, hdrb)
    # Check that comments have been merged
    assert list(hdr['COMMENT']) == ['A super useful comment', 'Another comment']
    assert list(hdr['HISTORY']) == ['This is historic', 'And not so useful']


def test_section():
    s = Section.from_string("[1:1024,1:512")
    assert tuple(s) == (0, 1024, 0, 512)
    assert s.asIRAFsection() == "[1:1024,1:512]"
    assert s.asslice() == [slice(0, 512), slice(0, 1024)]
    assert (s.x1, s.x2, s.y1, s.y2) == (0, 1024, 0, 512)
    assert (s[0], s[1], s[2], s[3]) == (0, 1024, 0, 512)

    s = Section.from_shape((8, 512, 1024))
    assert tuple(s) == (0, 1024, 0, 512, 0, 8)
    assert s.z2 == 8
