import os
import pytest
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats

import astrodata, gemini_instruments
from geminidr.gmos.primitives_gmos_longslit import GMOSLongslit
from gempy.library import peak_finding, wavecal

# NB N20210622S0320.fits is NOT the raw file; it has had its pixels
# replaced with those from the CS region of N20210622S0318.fits, to
# ensure the wavelength solutions are (almost) identical.

# These are actually the arcs as well, but that's OK
SCIENCE = ("N20210622S0318.fits",
           "N20210622S0320.fits",
          )
ARCS = ("N20210622S0318_arc.fits",
        "N20210622S0320_arc.fits",
       )


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("science", SCIENCE)
@pytest.mark.parametrize("arc", ARCS)
@pytest.mark.parametrize("save_and_reload", (True, False))
def test_spectral_wcs_stability(science, arc, save_and_reload,
                                path_to_inputs, change_working_dir):
    """
    This test checks that the combination of attachWavelengthSolution(),
    distortionCorrect(), and linearizeSpectra() work together, keeping the
    RA and Dec of the science image correct and keeping the wavelength
    solution (once it has been attached) correct. For this test, arc frames
    for the "science" so that the validity of the wavelength solution can
    be checked by finding peaks in the "science" frames and using the
    WCS-derived wavelengths.

    First is the comparision of unmosiacked data before and after
    attachWavelengthSolution. The RA and DEC should not change. No check
    is made on the wavelength, since this would be a comparison between the
    approximate linear solution and the true wavelength solution.

    Second, distortionCorrect() is run. Again, a check is made that the
    RA and DEC do not change. The wavelength is more complicated, since
    the Central Spectrum ROI is not in the middle of the Full Frame, and
    the distortion map preserves the wavelength solution in the middle
    row by default. However, the offset is only 12 rows and the curvature
    is small, so we check for agreement within a small fraction of a pixel.

    Finally, a check is made that linearizeSpectra can be run successfully,
    which will fail if the WAVE and SKY models are not readily separable.
    """

    def compare_peaks(ad, y, ref_peaks, fwidth=4, dw=1):
        """
        This function extracts a 1D spectrum from a 2D arc spectral image,
        finds the peaks, converts them to wavelengths based on the WCS of
        that image, and then compares these wavelengths with a reference
        list.

        Parameters
        ----------
        ad: AstroData instance
            Single-extension spectral image to measure peaks from
        y: int
            row number to extract 1D spectrum from
        ref_peaks: array
            list of wavelengths to which peaks should match
        fwidth: float
            width (in pixels) of features for peak-finding algorithm
        dw: float
            disperion (nm/pixel)
        """
        spectrum = peak_finding.average_along_slit(ad[0], center=y, nsum=10)
        peaks, _ = wavecal.find_line_peaks(*spectrum[:3], fwidth=fwidth, min_snr=5)
        wave_peaks = ad[0].wcs(peaks, y)[0]

        # Match peaks based on wavelengths, 3 pixel tolerance
        diffs = []
        for pp, wp in zip(peaks, wave_peaks):
            p0 = ref_peaks[np.argmin(abs(ref_peaks - wp))]
            if abs(wp - p0) < 3 * dw:
                diffs.append(wp - p0)

        # Systematic shifts are likely to be a full pixel, especially with a
        # Central Spectrum science and Full Frame arc, so we can have a modest
        # tolerance for an absolute shift here. Also check the number of
        # matches since it's possible that a large shift might match a few
        # lines incorrectly with a fortuitously small apparent shift.
        assert len(diffs) > 55
        mean, median, stddev = sigma_clipped_stats(diffs)
        assert abs(mean) < 0.1 * dw

    with change_working_dir():
        ad_sci = astrodata.from_file(os.path.join(path_to_inputs, science))
        p = GMOSLongslit([ad_sci])
        p.prepare()
        p.addDQ()
        p.addVAR(read_noise=True)
        p.overscanCorrect()
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True)
        if save_and_reload:
            p.writeOutputs()
            ad_sci = astrodata.from_file(ad_sci.filename)
            p = GMOSLongslit([ad_sci])

        ad_arc = astrodata.from_file(os.path.join(path_to_inputs, arc))
        wtable = ad_arc[0].WAVECAL
        arc_wave_peaks = wtable["wavelengths"]
        fwidth = wtable["coefficients"][list(wtable["name"]).index("fwidth")]
        y = ad_sci[4].shape[0] // 2
        dw = abs(ad_arc[0].wcs(0, 0)[0] - ad_arc[0].wcs(1, 0)[0])

        wcs_coords0 = ad_sci[4].wcs(0, y)
        sky0 = SkyCoord(*wcs_coords0[1:], unit=u.deg)
        if ad_sci[0].shape[0] > ad_arc[0].shape[0]:
            # Should refuse to run if arc is smaller than the science
            with pytest.raises(Exception):
                p.attachWavelengthSolution(arc=ad_arc)
            return
        else:
            p.attachWavelengthSolution(arc=ad_arc)
        wcs_coords1 = ad_sci[4].wcs(0, y)
        wave1 = wcs_coords1[0]
        sky1 = SkyCoord(*wcs_coords1[1:], unit=u.deg)
        assert sky1.separation(sky0) < 0.01 * u.arcsec

        ad_out = p.distortionCorrect()[0]

        compare_peaks(ad_out, y, arc_wave_peaks, fwidth=fwidth, dw=dw)

        p.linearizeSpectra()
        compare_peaks(ad_out, y, arc_wave_peaks, fwidth=fwidth, dw=dw)
