import numpy as np
import pytest

import os

from astropy.modeling import fitting, models
from astropy import units as u
from gwcs import coordinate_frames as cf
from gwcs.wcs import WCS as gWCS

import astrodata, gemini_instruments
from astrodata.testing import ad_compare

from geminidr.gnirs.primitives_gnirs_longslit import GNIRSLongslit
from geminidr.f2.primitives_f2_longslit import F2Longslit
from geminidr.core.primitives_telluric import make_linelist, GaussianLineSpreadFunction
from gempy.library import astromodels as am

from recipe_system.mappers.primitiveMapper import PrimitiveMapper


@pytest.fixture
def ext(request):
    npix = 20000  # actually there will be 1 more than this
    if request.param == "linear":
        wave_model = models.Scale(700 / npix) | models.Shift(300)
    elif request.param == "loglinear":
        wave_model = models.Exponential1D(amplitude=300, tau=npix/np.log(10/3))
    else:
        raise ValueError(f"Don't know {request.param}")
    flux = np.zeros((npix+1,), dtype=np.float32)

    # Put lines at 350, 450, ..., 950nm
    waves = wave_model(np.arange(flux.size))
    for linewave in range(350, 951, 100):
        flux[np.argmin(abs(linewave - waves))] = 1000

    ext = astrodata.NDAstroData(data=flux)
    input_frame = astrodata.wcs.pixel_frame(naxes=1)
    output_frame = cf.SpectralFrame(axes_order=(0,), unit=u.nm,
                                    axes_names=("WAVE",))
    ext.wcs = gWCS([(input_frame, wave_model),
                    (output_frame, None)])
    return ext


@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize("filename,mag,bbtemp",
                         [("hip93667_109_ad.fits", "K=5.241", 9650)]
                         )
def test_fit_telluric(path_to_inputs, path_to_refs, filename, mag, bbtemp):
    """
    Overall regression test for fitTelluric() with some parameters fixed
    """
    ad = astrodata.open(os.path.join(path_to_inputs, filename))

    pm = PrimitiveMapper(ad.tags, ad.instrument(generic=True).lower(),
                         mode='sq', drpkg='geminidr')
    pclass = pm.get_applicable_primitives()
    p = pclass([ad])
    adout = p.fitTelluric(magnitude=mag, bbtemp=bbtemp,
                          shift_tolerance=None,
                          debug_stellar_mask_threshold=0.).pop()

    adref = astrodata.open(os.path.join(path_to_refs, adout.filename))
    assert ad_compare(adout, adref)

    # Compare PCA coefficients
    assert np.allclose(adout.TELLFIT['PCA coefficients'].data,
                       adref.TELLFIT['PCA coefficients'].data)

    # Compare data-derived absorption
    for ext_out, ext_ref in zip(adout, adref):
        assert np.allclose(ext_out.TELLABS, ext_ref.TELLABS)

        # Compare evaluations of SENSFUNCs
        sensfunc_out = am.table_to_model(ext_out.SENSFUNC)
        sensfunc_ref = am.table_to_model(ext_ref.SENSFUNC)
        pixels = np.arange(ext_out.data.size)
        assert np.allclose(sensfunc_out(pixels), sensfunc_ref(pixels))


@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize("filename,magnitude,bbtemp,shift",
                         [("hip93667_109_ad.fits", "K=5.241", 9650, 0.18)])
def test_fit_telluric_xcorr(path_to_inputs, caplog, filename,
                            magnitude, bbtemp, shift):
    """
    Test of the cross-correlation in fitTelluric.

    The correct shift has been determined empirically using the GUI.
    """
    ad = astrodata.open(os.path.join(path_to_inputs, filename))

    pm = PrimitiveMapper(ad.tags, ad.instrument(generic=True).lower(),
                         mode='sq', drpkg='geminidr')
    pclass = pm.get_applicable_primitives()
    p = pclass([ad])
    adout = p.fitTelluric(magnitude=magnitude, bbtemp=bbtemp,
                          shift_tolerance=0).pop()

    for record in caplog.records:
        fields = record.message.split()
        if fields[0].lower() == "shift":
            assert float(fields[-2]) == pytest.approx(shift, abs=0.1)

    # Also check some things here to avoid writing another test that will
    # have to run the primitive again
    assert hasattr(adout, "TELLFIT")
    assert adout[0].data.dtype == np.float32
    assert adout[0].TELLABS.dtype == np.float32


@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize("filename,telluric,apply_model,shift",
                         [("N20171214S0147_extracted.fits", "N20171214S0163_telluric_selfwavecal.fits", True, 0),
                          ("N20171214S0147_extracted.fits", "N20171214S0163_telluric_selfwavecal.fits", False, 0),
                          ("N20171214S0147_extracted.fits", "N20171214S0163_telluric.fits", True, 0),
                          ("N20171214S0147_extracted.fits", "N20171214S0163_telluric.fits", False, -0.35),
                          ])
def test_telluric_correct_xcorr(path_to_inputs, caplog, filename, telluric,
                                apply_model, shift):
    """
    Test of the cross-correlation in telluricCorrect.

    The original dataset used here (N20171214S0147_extracted.fits) has a good
    wavelength solution. So the shift should be zero if apply_model=True,
    regardless of the telluric. Two tellurics are used: the "selfwavecal" one
    has a good wavelength solution, so the shift should be zero; the other one
    has a poor solution so a shift of -0.35 pixels should be found.
    """
    ad = astrodata.open(os.path.join(path_to_inputs, filename))

    pm = PrimitiveMapper(ad.tags, ad.instrument(generic=True).lower(),
                         mode='sq', drpkg='geminidr')
    pclass = pm.get_applicable_primitives()
    p = pclass([ad])
    adout = p.telluricCorrect(telluric=os.path.join(path_to_inputs, telluric),
                              shift_tolerance=0, apply_model=apply_model).pop()
    assert adout[0].data.dtype == np.float32

    for record in caplog.records:
        fields = record.message.split()
        if fields[0].lower() == "shift":
            assert float(fields[-2]) == pytest.approx(shift, abs=0.1)


@pytest.mark.parametrize("ext", ["linear", "loglinear"], indirect=True)
@pytest.mark.parametrize("resolution", [300, 500, 1000, 2000])
def test_gaussian_line_spread_function_convolve(ext, resolution):
    """Basic test that we convolve correctly"""
    lsf = GaussianLineSpreadFunction(ext, resolution=resolution)
    for w0 in range(350, 951, 100):
        wout, fout = lsf.convolve((w0-50, w0+50), lsf.all_waves, ext.data)
        m_init = models.Gaussian1D(amplitude=fout.max(), mean=w0)
        fit_it = fitting.TRFLSQFitter()
        m_final = fit_it(m_init, wout, fout)
        assert m_final.mean == pytest.approx(wout[fout.argmax()], abs=0.01)
        assert m_final.stddev * 2.35482 == pytest.approx(w0 / resolution,
                                                         rel=0.01)


@pytest.mark.parametrize("ext", ["linear", "loglinear"], indirect=True)
@pytest.mark.parametrize("resolution", [300, 500, 1000, 2000])
def test_gaussian_line_spread_function_convolve_and_resample(ext, resolution):
    """Basic test that we convolve and resample correctly"""
    lsf = GaussianLineSpreadFunction(ext, resolution=resolution)
    for w0 in range(350, 951, 100):
        wout = np.arange(w0-20, w0+20, 0.01)
        fout = lsf.convolve_and_resample(wout, lsf.all_waves, ext.data)
        m_init = models.Gaussian1D(amplitude=fout.max(), mean=w0)
        fit_it = fitting.TRFLSQFitter()
        m_final = fit_it(m_init, wout, fout)
        assert m_final.mean == pytest.approx(wout[fout.argmax()], abs=0.01)
        assert m_final.stddev * 2.35482 == pytest.approx(w0 / resolution,
                                                         rel=0.01)

@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize("filename,model_params",
                         # GNIRS L-band, sky emission
                         [('N20100820S0214_wavelengthSolutionDetermined.fits',
                           {"absorption": False, "num_lines": 100,
                            "resolution": 1759.0}),
                          # GNIRS J-band, sky absorption in object spectrum
                          ('N20121221S0199_aperturesFound.fits',
                           {"absorption": True, "num_lines": 50}),
                         ])
def test_get_atran_linelist(filename, model_params, change_working_dir,
                             path_to_inputs, path_to_refs):
    ad = astrodata.open(os.path.join(path_to_inputs, filename))
    p = GNIRSLongslit([])
    wave_model = am.get_named_submodel(ad[0].wcs.forward_transform, 'WAVE')
    linelist = p._get_atran_linelist(wave_model=wave_model, ext=ad[0],
                                     config=model_params)
    with change_working_dir(path_to_refs):
        linelist_fname = filename.split("_")[0] + "_atran_linelist.dat"
        ref_linelist = np.loadtxt(linelist_fname)

    np.testing.assert_allclose(linelist.vacuum_wavelengths(units="nm"),
                               ref_linelist, atol=1e-3)


@pytest.mark.parametrize("num_lines", (10, 50))
def test_make_linelist(num_lines):
    # Make a fake spectrum with 100 equally-spaced lines of increasing
    # strength and check that we recover the N//10 strongest in each of
    # 10 regions
    waves = np.arange(1000, 2000, 0.1)
    flux = np.zeros_like(waves)
    resolution = 2000
    sigma = 0.4 * waves.mean() / resolution
    lines = np.linspace(waves[0], waves[-1], 102)[1:-1]
    for i, w in enumerate(lines):
        flux += (i + 1) * 50 *np.exp(-0.5 * ((waves - w) / sigma) ** 2)

    linelist = make_linelist([waves, flux], resolution=resolution,
                             num_bins=10, num_lines=num_lines)
    expected_linelist = np.hstack([lines[i*10+10-num_lines // 10:i*10+10] for
                                   i in range(10)])
    assert len(linelist) == num_lines
    np.testing.assert_allclose(linelist[:, 0], expected_linelist)


@pytest.mark.preprocessed_data
@pytest.mark.regression
def test_get_airglow_linelist(path_to_inputs, path_to_refs):
    # Spectrum of F2 OH-emission sky lines for plotting
    # We use the _wavelengthSolutionDetermined file because thw WAVE model
    # is a Chebyshev1D, as required. (In normal reduction, a Cheb1D will be
    # provided bto _get_sky_spectrum() y determineWavelengthSolution,
    # regardless of the state of the input file.)
    ad_f2 = astrodata.open(os.path.join(
        path_to_inputs, 'S20180114S0104_wavelengthSolutionDetermined.fits'))
    wave_model = am.get_named_submodel(ad_f2[0].wcs.forward_transform, 'WAVE')

    p = F2Longslit([])
    linelist = p._get_airglow_linelist(wave_model=wave_model, ext=ad_f2[0],
                                       config={"absorption": False, "num_lines": 100,
                                               "resolution": 620.0})
    refplot_data_f2 = linelist.reference_spectrum
    ref_refplot_spec_f2 = np.loadtxt(
        os.path.join(path_to_refs, "S20180114S0104_refplot_spec.dat"))

    np.testing.assert_allclose(ref_refplot_spec_f2, refplot_data_f2["refplot_spec"], atol=1e-3)

