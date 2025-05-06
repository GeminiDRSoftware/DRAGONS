import pytest

import numpy as np
import astrodata, gemini_instruments
from astrodata.testing import download_from_archive
from geminidr.gnirs.primitives_gnirs_longslit import GNIRSLongslit

X = np.arange(1024)
Y = np.full_like(X, 511)  # row 511 is the slit

@pytest.mark.gnirsls
@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("filename", ("N20150108S0306.fits",
                                      "S20050117S0204.fits"))
def test_longslit_wcs(change_working_dir, filename):
    """
    Prepare a GNIRS longslit file and confirm that the sky coordinates
    along the slit have not changed.
    """
    with change_working_dir():
        file_path = download_from_archive(filename)
        ad = astrodata.open(file_path)
        p = GNIRSLongslit([ad])
        coords1 = ad[0].wcs(X, Y)
        p.prepare()
        coords2 = ad[0].wcs(X, Y)
        assert len(coords2) == 3
        np.testing.assert_allclose(coords1, coords2[1:], atol=1e-6)
        ad.write("test.fits", overwrite=True)
        ad2 = astrodata.open("test.fits")
        assert "FITS-WCS" not in ad2.phu  # not APPROXIMATE
        coords3 = ad[0].wcs(X, Y)
        assert len(coords3) == 3
        np.testing.assert_allclose(coords1, coords3[1:], atol=1e-6)
