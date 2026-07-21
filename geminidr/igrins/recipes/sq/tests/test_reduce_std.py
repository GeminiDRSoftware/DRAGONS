import os
import pytest

import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord

import astrodata, gemini_instruments
from astrodata.testing import ad_compare
from recipe_system.reduction.coreReduce import Reduce


# tuples with list of input files, dict of calibrations
STD_INPUTS = [
    ([f"N20260303S{i:04d}_K.fits" for i in range(28, 32)], {"processed_flat": "N20260228S0543_K_flat.fits",
                                                            "processed_arc": "N20260301S0028_K_arc.fits"})
]


@pytest.fixture()
def input_files(request, path_to_inputs):
    return [os.path.join(path_to_inputs, filename)
            for filename in request.param]


@pytest.mark.skip("Fails now due to a dependency update?")
@pytest.mark.igrins2
@pytest.mark.preprocessed_data
@pytest.mark.parametrize('input_files, caldict', STD_INPUTS, indirect=['input_files'])
def test_make_processed_std(input_files, caldict, change_working_dir, path_to_inputs,
                            path_to_refs):
    r = Reduce()
    r.files = input_files
    # This avoids issues when running locally since test_make_processed_bpm
    # will add the BPM to the caldb
    r.uparms = {'addDQ:static_bpm': None}
    r.ucals = {k : os.path.join(path_to_inputs, v) for k, v in caldict.items()}
    with change_working_dir():
        r.runr()
        assert r.recipename == "makeStd"
        output_filename = r._output_filenames.pop()
        adout = astrodata.open(output_filename)
        adref = astrodata.open(os.path.join(path_to_refs, output_filename))
        ad_compare(adout, adref, ignore_kw=[k for k in adref[0].hdr['WAT2*']] + ["ADDMDF", "SDZWCS"])
        assert_allclose(adref[0].WAVELENGTHS, adout[0].WAVELENGTHS)

        output_filename = output_filename.replace("1d", "2d")
        adout = astrodata.open(output_filename)
        adref = astrodata.open(os.path.join(path_to_refs, output_filename))
        ad_compare(adout, adref, ignore_kw=[k for k in adref[0].hdr['WAT2*']] + ["ADDMDF", "SDZWCS"])
        assert_allclose(adref[0].WAVELENGTHS, adout[0].WAVELENGTHS)

        output_filename = output_filename.replace("2d", "_debug")
        adout = astrodata.open(output_filename)
        adref = astrodata.open(os.path.join(path_to_refs, output_filename))
        ad_compare(adout, adref, ignore=["wcs"], ignore_kw=[k for k in adref[0].hdr['WAT2*']] + ["ADDMDF", "SDZWCS"])


@pytest.mark.igrins2
@pytest.mark.preprocessed_data
@pytest.mark.parametrize('input_files, caldict', STD_INPUTS, indirect=['input_files'])
def test_new_make_processed_std(input_files, caldict, change_working_dir, path_to_inputs):
    r = Reduce()
    r.recipename = "makeStellarNew"
    r.files = input_files
    # This avoids issues when running locally since test_make_processed_bpm
    # will add the BPM to the caldb
    r.uparms = {'addDQ:static_bpm': None,
                'attachWavelengthSolution:write_outputs': True}
    r.ucals = {k : os.path.join(path_to_inputs, v.replace(".fits", "_dragons.fits"))
               for k, v in caldict.items()}
    with change_working_dir():
        # I'm only interested in running the recipe as far as attachWavelengthSolution
        try:
            r.runr()
        except:
            pass
        adout = astrodata.open(os.path.split(input_files[0])[1].replace(
            '_K', '_K_wavelengthSolutionAttached'))

        assert len(adout) == 24
        np.testing.assert_equal(adout.hdr['SPECORDR'], list(range(70, 94)))

        # WCS should be (wavelength, RA, dec) with real sky coordinates,
        for ext in adout:
            ymid = ext.shape[0] // 2
            wcs1 = ext.wcs(1024, ymid)
            assert len(wcs1) == 3
            slitlen_pix = np.diff(ext.SLITEDGE['c0'] + ext.SLITEDGE['c2'])[0]
            c1 = SkyCoord(*wcs1[1:], unit="deg")
            c2 = SkyCoord(*ext.wcs(1024, ymid+1)[1:], unit="deg")
            assert c1.separation(c2).arcsec == pytest.approx(5. / slitlen_pix, rel=0.05)

            # The "distortion_corrected" frame should be in uniform pixels,
            # but the input pixels are not uniform because the slit width
            # in pixels varies across the detector.
            t = ext.wcs.get_transform('pixels', 'distortion_corrected')
            assert t(1024, ymid+1)[1] - t(1024, ymid)[1] == pytest.approx(50. / slitlen_pix, rel=0.05)
