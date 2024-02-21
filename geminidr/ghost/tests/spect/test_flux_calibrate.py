# test for fluxCalibrate(), that we get the same sort of result
# for different binnings of the standard

# This test needs to be improved once I sort out the GHOST fluxcal

import pytest
import os

import numpy as np

import astrodata, gemini_instruments
from gempy.library import astrotools as at

from geminidr.ghost.primitives_ghost_spect import GHOSTSpect, make_wavelength_table

# These should come from identical raw files where one has been binned in software
DATASETS = [("redunbinned_sensitivityCalculated.fits",
             "redbinned_sensitivityCalculated.fits")]


@pytest.mark.ghostspect
@pytest.mark.parametrize("std_filenames", DATASETS)
def test_flux_calibrate_binning(path_to_inputs, std_filenames):
    # We calibrate the standards using each other to confirm
    # things work when the standard has a lower and a higher
    # binning. The criterion is a bit empirical because we don't
    # get exactly the same result when binning. We ignore the
    # most extreme orders (lowest S/N) and look at the 95th
    # percentile deviation
    for sci_filename in std_filenames:
        outputs = {}
        for std_filename in std_filenames:
            ad_sci = astrodata.open(os.path.join(path_to_inputs, sci_filename))
            arm = ad_sci.arm()
            sci_xbin, sci_ybin = ad_sci.detector_x_bin(), ad_sci.detector_y_bin()
            p = GHOSTSpect([ad_sci])
            ad_std = astrodata.open(os.path.join(path_to_inputs, std_filename))
            xbin, ybin = ad_std.detector_x_bin(), ad_std.detector_y_bin()
            outputs[(xbin, ybin)] = p.fluxCalibrate(standard=ad_std).pop()

        # Ensure all outputs are the same shape
        assert len(set(v[0].shape for v in outputs.values())) == 1

        base_output = outputs[(sci_xbin, sci_ybin)]
        w = make_wavelength_table(base_output[0])
        rms_values = [at.std_from_pixel_variations(row)
                      for row in base_output[0].data]

        data = {k: v[0].data for k, v in outputs.items()}
        for k, v in data.items():
            bad_fom = []
            if k != (sci_xbin, sci_ybin):
                for i in range(w.shape[0]):
                    # ignore extrme orders
                    if (arm == "red" and i < 2) or (arm == "blue" and w.shape[0] - i < 3):
                        continue
                    dev = v[i] - base_output[0].data[i]
                    fom = np.percentile(abs(dev) / rms_values[i], 95)
                    if fom > 1:
                        bad_fom.append((w[i].min(), fom))
        assert not bad_fom, bad_fom

