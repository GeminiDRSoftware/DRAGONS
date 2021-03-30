import os
from glob import glob
import pytest

import numpy as np
from astropy.table import Table

import astrodata, gemini_instruments
from geminidr.gmos.primitives_gmos_longslit import GMOSLongslit

input_files = ["S20190206S0108_fluxCalibrated.fits"]
formats = [("ascii", "dat", "ascii.basic"),
           ("fits", "fits", "fits"),
           ("csv", "csv", "ascii.csv")]

@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad", input_files, indirect=True)
@pytest.mark.parametrize("output_format, extension, input_format", formats)
def test_write_spectrum(ad, output_format, extension, input_format, change_working_dir):

    with change_working_dir():
        nfiles = len(glob("*"))
        p = GMOSLongslit([ad])
        p.write1DSpectra(apertures=1, format=output_format,
                         extension=extension)
        assert len(glob("*")) == nfiles + 1
        t = Table.read(ad.filename.replace(".fits", "_001.dat"),
                       format=input_format)
        assert len(t) == ad[0].data.size
        np.testing.assert_allclose(t["data"].data, ad[0].data, atol=1e-9)
        p.write1DSpectra(apertures=None, format=output_format,
                         extension=extension, overwrite=True)
        assert len(glob("*")) == nfiles + len(ad)


# Local Fixtures and Helper Functions ------------------------------------------
@pytest.fixture(scope='function')
def ad(path_to_inputs, request):
    """
    Returns the pre-processed spectrum file.

    Parameters
    ----------
    path_to_inputs : pytest.fixture
        Fixture defined in :mod:`astrodata.testing` with the path to the
        pre-processed input file.
    request : pytest.fixture
        PyTest built-in fixture containing information about parent test.

    Returns
    -------
    AstroData
        Input spectrum processed up to right before the
        `determineWavelengthSolution` primitive.
    """
    filename = request.param
    path = os.path.join(path_to_inputs, filename)

    if os.path.exists(path):
        ad = astrodata.open(path)
    else:
        raise FileNotFoundError(path)

    return ad
