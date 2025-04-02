#!/usr/bin/env python
"""
Tests for the p.addOIWFStoDQ primitive.
"""
import logging
import pytest
import re
import astrodata
import gemini_instruments

from astrodata.testing import download_from_archive
from geminidr.gmos.primitives_gmos_image import GMOSImage
from gempy.utils import logutils


@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("filename", ["N20190102S0162.fits"])
def test_oiwfs_not_used_in_observation(caplog, filename):
    """
    Test that nothing happens when the input file does not use the OIWFS.

    Parameters
    ----------
    caplog : fixture
    filename : str
    """
    caplog.set_level(logging.DEBUG)
    file_path = download_from_archive(filename)
    ad = astrodata.from_file(file_path)

    p = GMOSImage([ad])
    p.addOIWFSToDQ()

    print(caplog.records)
    assert len(caplog.records) > 0
    assert any("OIWFS not used for image" in r.message for r in caplog.records)


@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("filename", ["N20190101S0051.fits"])
def test_warn_if_dq_does_not_exist(caplog, filename):
    """
    Test that the primitive does not run if the input file does not have a DQ
    plan.

    Parameters
    ----------
    caplog : fixture
    filename : str
    """
    caplog.set_level(logging.DEBUG)
    file_path = download_from_archive(filename)
    ad = astrodata.from_file(file_path)

    p = GMOSImage([ad])
    p.addOIWFSToDQ()

    assert len(caplog.records) > 0
    assert any("No DQ plane for" in r.message for r in caplog.records)


@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("filename, x0, y0, ext_num",
                         [("S20190105S0168.fits", 130, 483, 9)])
def test_add_oiwfs_runs_normally(caplog, ext_num, filename, x0, y0):
    """
    Test that the primitive does not run if the input file does not have a DQ
    plan.

    Parameters
    ----------
    caplog : fixture
    filename : str
    """
    caplog.set_level(logging.DEBUG)
    file_path = download_from_archive(filename)
    ad = astrodata.from_file(file_path)

    p = GMOSImage([ad])
    p.addDQ()
    p.addVAR(read_noise=True)
    p.addOIWFSToDQ()

    # plot(p.streams['main'][0])
    assert len(caplog.records) > 0
    assert any("Guide star location found at" in r.message for r in caplog.records)

    # Some kind of regression test
    for r in caplog.records:
        if r.message.startswith("Guide star location found at"):
            coords = re.findall(r"\((.*?)\)", r.message).pop().split(',')

            x = float(coords[0])
            y = float(coords[1])
            n = int(r.message.split(' ')[-1])

            assert abs(x - x0) < 1
            assert abs(y - y0) < 1
            assert n == ext_num


@pytest.mark.skip("Test fails on Jenkins - Investigate why")
@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("filename", ["N20190101S0051.fits"])
def test_add_oiwfs_warns_when_wfs_if_not_in_field(caplog, filename):
    """
    Test that the primitive does not run if the input file does not have a DQ
    plan.

    Parameters
    ----------
    caplog : fixture
    filename : str
    """
    caplog.set_level(logging.DEBUG)
    file_path = download_from_archive(filename)
    ad = astrodata.from_file(file_path)

    p = GMOSImage([ad])
    p.addDQ()
    p.addVAR(read_noise=True)
    p.addOIWFSToDQ()

    # plot(p.streams['main'][0])
    print(caplog.records)
    assert any("No good rows in" in r.message for r in caplog.records)

    assert any("Cannot distinguish probe region from sky for"
               in r.message for r in caplog.records)


# -- Helper functions --------------------------------------------------------
def plot(ad):
    """
    Displays the tiled arrays with the DQ mask for analysing the data.

    Parameters
    ----------
    ad : multi-extension data
    """
    from astropy.visualization import ImageNormalize, ZScaleInterval
    from copy import deepcopy
    import numpy as np
    import matplotlib.pyplot as plt

    p = GMOSImage([deepcopy(ad)])
    _ad = p.tileArrays().pop()

    fig, axs = plt.subplots(num=ad.filename, ncols=len(_ad), sharey=True)

    norm = ImageNormalize(
        np.concatenate([ext.data.ravel()[ext.mask.ravel() == 0] for ext in _ad]),
        interval=ZScaleInterval())

    vmin = norm.vmin
    vmax = norm.vmax

    for i, ext in enumerate(_ad):

        data = np.ma.masked_array(ext.data, mask=ext.mask)
        cmap = plt.get_cmap('viridis')
        cmap.set_bad('red', alpha='0.5')
        axs[i].imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        # axs[i].imshow(data.data, origin='lower', vmin=vmin, vmax=vmax)

    plt.show()


if __name__ == '__main__':
    pytest.main()
