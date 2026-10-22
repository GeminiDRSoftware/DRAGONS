import pytest
import os

import numpy as np
from astropy.coordinates import SkyCoord

import astrodata, gemini_instruments
from geminidr.gnirs.primitives_gnirs_crossdispersed import GNIRSCrossDispersed


INPUT_FILES = [
    ("N20210101S0030_flatCorrected.fits", "N20210101S0065_arc.fits")
               ]


@pytest.mark.gnirsxd
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("sci, arc", INPUT_FILES)
def test_attach_wavelength_solution(sci, arc, path_to_inputs):
    """Check tht the attachWavelengthSolution primitive works as expected.
    There's no reference file for this test; instead checks are made on
    the general properties of the output's WCS"""
    ad = astrodata.open(os.path.join(path_to_inputs, sci))
    adarc = astrodata.open(os.path.join(path_to_inputs, arc))
    p = GNIRSCrossDispersed([ad])
    p.attachWavelengthSolution(arc=adarc)
    adout = p.writeOutputs().pop()

    for ext, ext_arc in zip(adout, adarc):
        assert ext.wcs.available_frames == ext_arc.wcs.available_frames

        Y, X = np.mgrid[:ext.shape[0], :ext.shape[1]]
        xx, yy = X[ext.mask == 0], Y[ext.mask == 0]

        # Confirm that wavelengths of each pixel in the science are the
        # same as in the arc
        world, world_arc = ext.wcs(xx, yy), ext_arc.wcs(xx, yy)
        np.testing.assert_allclose(world[0], world_arc[0], atol=1e-4)

        # Confirm the roundtrip
        world = ext.wcs(xx, yy)
        roundtrip = ext.wcs.invert(*world)
        np.testing.assert_allclose(roundtrip[0], xx, atol=0.01)
        np.testing.assert_allclose(roundtrip[1], yy, atol=0.01)

        # Confirm the plate scale. This has a large tolerance because the
        # slits need not be of constant width in on the detector.
        xmid = 0.5 * ext.shape[1]
        ypixels = np.arange(ext.shape[0])
        sky1 = SkyCoord(*ext.wcs([xmid] * ypixels.size, ypixels)[1:],
                        unit="deg")
        sky2 = SkyCoord(*ext.wcs([xmid + 1] * ypixels.size, ypixels)[1:],
                        unit="deg")
        for c1, c2 in zip(sky1, sky2):
            assert c1.separation(c2).arcsec == pytest.approx(ext.pixel_scale(),
                                                             rel=0.05)


@pytest.mark.gnirsxd
@pytest.mark.preprocessed_data
def test_attach_wavelength_solution_missing_distortion_models(path_to_inputs):
    """
    Simple test to check that the attachWavelengthSolution primitive works if
    extensions are missing distortion models.
    """
    ad = astrodata.open(os.path.join(path_to_inputs, "N20200818S0038_flatCorrected.fits"))
    arc = astrodata.open(os.path.join(path_to_inputs, "N20200818S0350_arc.fits"))

    p = GNIRSCrossDispersed([ad])
    p.attachWavelengthSolution(arc=arc)

    num_ext_without_distortion = 0
    for ext, arc_ext in zip(p.adinputs[0], arc):
        assert (("distortion_corrected" in ext.wcs.available_frames) ==
                ("distortion_corrected" in arc_ext.wcs.available_frames))
        if "distortion_corrected" not in ext.wcs.available_frames:
            num_ext_without_distortion += 1

    # Because this test is specifically for the case where some extensions
    # are missing distortion models!
    assert num_ext_without_distortion > 0, "At least one extension should be missing distortion model"
