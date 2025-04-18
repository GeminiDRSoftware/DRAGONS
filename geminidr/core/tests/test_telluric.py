import numpy as np
import pytest

import os

import astrodata, gemini_instruments
from astrodata.testing import ad_compare

from geminidr.gnirs.primitives_gnirs_longslit import GNIRSLongslit
from geminidr.core.primitives_telluric import make_linelist
from gempy.library import astromodels as am

from recipe_system.mappers.primitiveMapper import PrimitiveMapper


@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize("filename", ["hip93667_109_ad.fits"])
def test_fit_telluric(path_to_inputs, path_to_refs, filename):
    """
    Overall regression test for fitTelluric() with some parameters fixed
    """
    ad = astrodata.open(os.path.join(path_to_inputs, filename))

    pm = PrimitiveMapper(ad.tags, ad.instrument(generic=True).lower(),
                         mode='sq', drpkg='geminidr')
    pclass = pm.get_applicable_primitives()
    p = pclass([ad])
    adout = p.fitTelluric(magnitude="K=5.241", bbtemp=9650,
                          shift_tolerance=None).pop()

    adref = astrodata.open(os.path.join(path_to_refs, adout.filename))
    assert ad_compare(adout, adref)

    # Compare PCA coefficients
    assert np.allclose(adout.TELLFIT['PCA coefficients'].data,
                       adref.TELLFIT['PCA coefficients'].data)

    # Compare data-derived absorption
    assert np.allclose(adout[0].TELLABS, adref[0].TELLABS)

    # Compare evaluations of SENSFUNCs
    sensfunc_out = am.table_to_model(adout[0].SENSFUNC)
    sensfunc_ref = am.table_to_model(adref[0].SENSFUNC)
    pixels = np.arange(adout[0].data.size)
    assert np.allclose(sensfunc_out(pixels), sensfunc_ref(pixels))


@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize("filename,shift", [("hip93667_109_ad.fits", 0.18)])
def test_fit_telluric_xcorr(path_to_inputs, caplog, filename, shift):
    """
    Test of the cross-correlation in fitTelluric
    """
    ad = astrodata.open(os.path.join(path_to_inputs, filename))

    pm = PrimitiveMapper(ad.tags, ad.instrument(generic=True).lower(),
                         mode='sq', drpkg='geminidr')
    pclass = pm.get_applicable_primitives()
    p = pclass([ad])
    adout = p.fitTelluric(magnitude="K=5.241", bbtemp=9650,
                          shift_tolerance=0)

    for record in caplog.records:
        fields = record.message.split()
        if fields[0].lower() == "shift":
            assert float(fields[-2]) == pytest.approx(shift, abs=0.1)


@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize("filename,model_params",
                         # GNIRS L-band, sky emission
                         [('N20100820S0214_wavelengthSolutionDetermined.fits',
                           {"absorption": False, "nlines": 100}),
                          # GNIRS J-band, sky absorption in object spectrum
                          ('N20121221S0199_aperturesFound.fits',
                           {"absorption": True, "nlines": 50}),
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
