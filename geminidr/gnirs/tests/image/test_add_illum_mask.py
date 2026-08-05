import pytest

import os
import numpy as np

from astrodata.testing import download_from_archive
import astrodata, gemini_instruments
from geminidr.gnirs.primitives_gnirs_image import GNIRSImage


# filename and (x, y) coords (0-indexed) of lowest unmasked pixel
DATASETS = (("N20110627S0031.fits", (625, 445)),  # ShortBlue, Wings, off-centre
            ("N20110627S0069.fits", (501, 386)),  # ShortBlue, Wings
            ("N20131222S0064.fits", (506, 370)),  # ShortBlue, NoWings
            ("N20140717S0229.fits", (540, 210)),  # LongBlue, Wings
            ("N20151118S0399.fits", (544, 233)),  # LongBlue, NoWings
            ("N20110106S0257.fits", (270, 180)),  # LongBlue, Wings, off-centre
            ("N20110227S0147.fits", (522, 380)),  # ShortRed, Wings
            ("N20220718S0078.fits", (452, 215)),  # LongRed, Wings (PROPRIETARY)
            )


@pytest.mark.gnirs
@pytest.mark.dragons_remote_data
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("filename,result", DATASETS)
def test_add_illum_mask(filename, result, change_working_dir, path_to_inputs):
    if filename.startswith("N2022"):
        ad = astrodata.open(os.path.join(path_to_inputs, filename))
    else:
        ad = astrodata.open(download_from_archive(filename))
    with change_working_dir():
        p = GNIRSImage([ad])
        p.prepare()  # bad_wcs="ignore")
        adout = p.addIllumMaskToDQ().pop()
        y, x = np.unravel_index(adout[0].mask.argmin(), adout[0].shape)
        # Get the middle unmasked pixel on the bottom row
        x = (np.arange(adout[0].shape[1])[adout[0].mask[y] == 0]).mean()
        assert abs(y - result[1]) <= 2
        assert abs(x - result[0]) <= 10
