#!/usr/bin/env python
"""
Tests for the p.addOIWFStoDQ primitive.
"""

import pytest

import astrodata
import gemini_instruments

from astrodata.testing import download_from_archive
from geminidr.gmos.primitives_gmos_image import GMOSImage


@pytest.mark.parametrize("filename", ["N20190102S0162.fits"])
def test_oiwfs_not_used_in_observation(caplog, filename):
    """
    Test that nothing happens when the input file does not use the OIWFS.

    Parameters
    ----------
    caplog : fixture
    filename : str
    """
    file_path = download_from_archive(filename)
    ad = astrodata.open(file_path)

    p = GMOSImage([ad])
    p.addOIWFSToDQ()

    assert any("OIWFS not used for image" in r.message for r in caplog.records)


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
    file_path = download_from_archive(filename)
    ad = astrodata.open(file_path)

    p = GMOSImage([ad])
    p.addOIWFSToDQ()

    assert any("No DQ plane for" in r.message for r in caplog.records)


@pytest.mark.parametrize("filename", ["S20190105S0168.fits"])
def test_add_oiwfs_runs_normally(caplog, filename):
    """
    Test that the primitive does not run if the input file does not have a DQ
    plan.

    Parameters
    ----------
    caplog : fixture
    filename : str
    """
    file_path = download_from_archive(filename)
    ad = astrodata.open(file_path)

    p = GMOSImage([ad])
    p.addDQ()
    p.addVAR(read_noise=True)
    p.addOIWFSToDQ()

    # plot(p.streams['main'][0])


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
    file_path = download_from_archive(filename)
    ad = astrodata.open(file_path)

    p = GMOSImage([ad])
    p.addDQ()
    p.addVAR(read_noise=True)
    p.addOIWFSToDQ()

    assert any("No good rows in" in r.message for r in caplog.records)

    assert any("Cannot distinguish probe region from sky for"
               in r.message for r in caplog.records)

    # plot(p.streams['main'][0])


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


def create_inputs():
    pass


if __name__ == '__main__':
    from sys import argv
    if '--create-inputs' in argv:
        create_inputs()
    else:
        pytest.main()
