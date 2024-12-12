"""
This test loads data from multi-extension instruments and checks that the
WCS created by prepare is good, and that there is stability during the
tiling and mosaicking operations, and reading and writing.

This is only implemented for GMOS at the present time.
"""
import pytest
import os

from astropy import units as u
from astropy.coordinates import SkyCoord
from importlib import import_module

import astrodata
from astrodata.testing import download_from_archive

from geminidr.gmos.primitives_gmos_image import GMOSImage

from gempy.library.transform import find_reference_extension

TEMPFILE = "wcs_test.fits"

# Full Frame, Central Spectrum, Central Stamp
GMOS_FILES = ["N20180119S0156.fits", "N20180113S0128.fits", "N20180115S0238.fits"]


@pytest.fixture(scope="module", params=GMOS_FILES)
def raw_ad_path(request):
    """

    Parameters
    ----------
    filename: str
        name of the file to be processed

    Returns
    -------

    """
    full_path = download_from_archive(request.param)
    return full_path


@pytest.fixture(params=[False, True])
def do_prepare(request):
    return request.param

@pytest.fixture(params=[False, True])
def do_overscan_correct(request):
    return request.param

@pytest.fixture(params=[False, True])
def tile_all(request):
    return request.param

def test_gmos_wcs_stability(raw_ad_path, do_prepare, do_overscan_correct, tile_all):
    raw_ad = astrodata.open(raw_ad_path)

    # Ensure it's tagged IMAGE so we can get an imaging WCS and can use SkyCoord
    raw_ad.phu['GRATING'] = 'MIRROR'

    # Check the reference extension is what we think and find the middle
    ref_index = find_reference_extension(raw_ad)
    assert ref_index == (len(raw_ad) - 1) // 2  # works for GMOS
    y, x = [length // 2 for length in raw_ad[ref_index].shape]
    c0 = SkyCoord(*raw_ad[ref_index].wcs(x, y), unit="deg")

    p = GMOSImage([raw_ad])  # TODO: support for other instruments
    geotable = import_module('.geometry_conf', p.inst_lookups)
    chip_gaps = geotable.tile_gaps[raw_ad.detector_name()]

    # Test that prepare keeps the reference extenion's WCS intact
    if do_prepare:
        p.prepare()
        c = SkyCoord(*raw_ad[ref_index].wcs(x, y), unit="deg")
        assert c0.separation(c) < 1e-10 * u.arcsec

    # Test that slicing the NDData keeps the WCS valid
    if do_overscan_correct:
        xshift, _, yshift, _ = raw_ad[ref_index].data_section()
        p.overscanCorrect()
        x -= xshift
        y -= yshift
        c = SkyCoord(*raw_ad[ref_index].wcs(x, y), unit="deg")
        assert c0.separation(c) < 1e-10 * u.arcsec

    # Test that tiling doesn't affect the reference extension's WCS
    new_ref_index = 0 if (tile_all or raw_ad.detector_roi_setting() == 'Central Stamp') else 1
    p.tileArrays(tile_all=tile_all)
    ad = p.streams['main'][0]

    first = 0 if tile_all else (len(raw_ad) // 3)  # index of first raw extension
    # These extension widths are either overscan-trimmed or not, as
    # required and so no alternative logic is required
    x += sum([ext.shape[1] for ext in raw_ad[first:ref_index]])
    if tile_all and raw_ad.detector_roi_setting() != 'Central Stamp':
        x += chip_gaps // raw_ad.detector_x_bin()
    c = SkyCoord(*ad[new_ref_index].wcs(x, y), unit="deg")
    assert c0.separation(c) < 1e-10 * u.arcsec

    # Now write the file to disk and read it back in and check WCS stability
    ad.write(TEMPFILE, overwrite=True)
    ad = astrodata.open(TEMPFILE)

    c = SkyCoord(*ad[new_ref_index].wcs(x, y), unit="deg")
    assert c0.separation(c) < 1e-9 * u.arcsec

    os.remove(TEMPFILE)
