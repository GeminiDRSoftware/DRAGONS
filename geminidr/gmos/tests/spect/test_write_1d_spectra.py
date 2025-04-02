import os
from glob import glob
import logging
import pytest

import numpy as np
from astropy.table import Table
from astropy import units as u

import astrodata, gemini_instruments
from geminidr.gmos.primitives_gmos_longslit import GMOSLongslit

input_files = ["S20190206S0108_extracted.fits",
               "S20190206S0108_fluxCalibrated.fits"]
formats = [("ascii", "dat", "ascii.basic"),
           ("fits", "fits", "fits"),
           ("ascii.csv", "csv", "ascii.csv")]


@pytest.mark.gmosls  # stop it running as a unit test
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad", input_files, indirect=True)
@pytest.mark.parametrize("output_format, extension, input_format", formats)
def test_write_spectrum_formats(ad, output_format, extension, input_format,
                                change_working_dir):
    """Test writing with various formats"""
    with change_working_dir():
        nfiles = len(glob("*"))
        p = GMOSLongslit([ad])
        p.write1DSpectra(apertures=1, format=output_format,
                         extension=extension)
        assert len(glob("*")) == nfiles + 1
        t = Table.read(ad.filename.replace(".fits", f"_001.{extension}"),
                       format=input_format)
        assert len(t) == ad[0].data.size
        np.testing.assert_allclose(t["data"].data, ad[0].data, atol=1e-9)
        p.write1DSpectra(apertures=None, format=output_format,
                         extension=extension, overwrite=True)
        assert len(glob("*")) == nfiles + len(ad)


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad", input_files, indirect=True)
@pytest.mark.parametrize("xunits", (None, "nm", "AA"))
@pytest.mark.parametrize("yunits", (None, "electron", "electron/s", "W/m^2/AA", "erg/cm^2/s/Hz"))
def test_write_spectrum_various_units(ad, xunits, yunits, change_working_dir, caplog):
    """Test writing with changes of units"""
    with change_working_dir():
        log = logging.getLogger('geminidr')
        log.warning('Emit a fake warning to reset DuplicateWarningFilter')
        p = GMOSLongslit([ad])
        w0 = ad[0].wcs(0)
        mismatched_units = (yunits is not None and
                            (('electron' in yunits and 'flux' in ad.filename) or
                             ('m^2' in yunits and 'flux' not in ad.filename)))
        p.write1DSpectra(apertures=1, wave_units=xunits, data_units=yunits,
                         var=True, overwrite=True)
        unit_conversion_failure = any("Cannot convert spectrum" in r.message
                                      for r in caplog.records)
        assert mismatched_units == unit_conversion_failure
        t = Table.read(ad.filename.replace(".fits", "_001.dat"),
                       format="ascii.basic")
        assert len(t) == ad[0].data.size
        assert len(t.colnames) == 3
        # Confirm wavelength has been converted
        assert t['wavelength'][0] == pytest.approx(
            (w0 * 10) if xunits == 'AA'  else w0)
        # Confirm S/N is preserved
        np.testing.assert_allclose(np.sqrt(ad[0].variance) / ad[0].data,
                                   np.sqrt(t['variance']) / t['data'], rtol=2e-7)



# Local Fixtures and Helper Functions ------------------------------------------
@pytest.fixture(scope='function')
def ad(request, path_to_inputs):
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
        ad = astrodata.from_file(path)
    else:
        raise FileNotFoundError(path)

    return ad
