import numpy as np

from geminidr.core import primitives_spect
from astrodata.testing import ad_compare
from geminidr.core.tests.test_spect import create_zero_filled_fake_astrodata, fake_point_source_spatial_profile, \
    get_aperture_table, SkyLines
from geminidr.interactive import server
import gemini_instruments


def interactive_test(tst):
    def set_test_mode():
        save_itm = server.test_mode
        server.test_mode = True
        tst()
        server.test_mode = save_itm
    return set_test_mode

import pytest

@pytest.mark.skip(reason="interactive tests not yet implemented")

# @interactive_test
# def test_trace_apertures_interactive():
#     print("starting test_trace_apertures_interactive")
#     # Input parameters ----------------
#     width = 400
#     height = 200
#     trace_model_parameters = {'c0': height // 2, 'c1': 5.0, 'c2': -0.5, 'c3': 0.5}
#
#     # Boilerplate code ----------------
#     def make_test_ad_p():
#         ad = create_zero_filled_fake_astrodata(height, width)
#         ad[0].data += fake_point_source_spatial_profile(height, width, trace_model_parameters)
#         ad[0].APERTURE = get_aperture_table(height, width)
#         return ad, primitives_spect.Spect([])
#
#     # Running the test ----------------
#     ad, _p = make_test_ad_p()
#     ad_out = _p.traceApertures([ad], order=len(trace_model_parameters) + 1)[0]
#
#     iad, _p = make_test_ad_p()
#     iad_out = _p.traceApertures([iad], order=len(trace_model_parameters) + 1, interactive=True)[0]
#
#     assert(ad_compare(ad_out, iad_out))
#
#
# @interactive_test
# def test_sky_correct_from_slit_interactive():
#     # Input Parameters ----------------
#     width = 200
#     height = 100
#     n_sky_lines = 50
#
#     # Simulate Data -------------------
#     np.random.seed(0)
#
#     source_model_parameters = {'c0': height // 2, 'c1': 0.0}
#
#     source = fake_point_source_spatial_profile(
#         height, width, source_model_parameters, fwhm=0.05 * height)
#
#     sky = SkyLines(n_sky_lines, width - 1)
#
#     def make_test_ad_p():
#         ad = create_zero_filled_fake_astrodata(height, width)
#         ad[0].data += source
#         ad[0].data += sky(ad[0].data, axis=1)
#
#         return ad, primitives_spect.Spect([])
#
#     ad, _p = make_test_ad_p()
#     ad_out = _p.skyCorrectFromSlit([ad], function="spline3", order=5,
#                                    grow=2, niter=3, lsigma=3, hsigma=3,
#                                    aperture_growth=2, interactive=False)[0]
#
#     iad, _p = make_test_ad_p()
#     iad_out = _p.skyCorrectFromSlit([iad], function="spline3", order=5,
#                                     grow=2, niter=3, lsigma=3, hsigma=3,
#                                     aperture_growth=2, interactive=True)[0]
#
#     assert(ad_compare(ad_out, iad_out))
#
#
# @interactive_test
# def test_sky_correct_from_slit_with_aperture_table_interactive():
#     # Input Parameters ----------------
#     width = 200
#     height = 100
#     n_sky_lines = 50
#
#     # Simulate Data -------------------
#     np.random.seed(0)
#
#     source_model_parameters = {'c0': height // 2, 'c1': 0.0}
#
#     source = fake_point_source_spatial_profile(
#         height, width, source_model_parameters, fwhm=0.08 * height)
#
#     sky = SkyLines(n_sky_lines, width - 1)
#
#     def make_test_ad_p():
#         ad = create_zero_filled_fake_astrodata(height, width)
#         ad[0].data += source
#         ad[0].data += sky(ad[0].data, axis=1)
#         ad[0].APERTURE = get_aperture_table(height, width)
#
#         return ad, primitives_spect.Spect([])
#
#     ad, p = make_test_ad_p()
#     ad_out = p.skyCorrectFromSlit([ad], function="spline3", order=5,
#                                   grow=2, niter=3, lsigma=3, hsigma=3,
#                                   aperture_growth=2, interactive=False)[0]
#
#     iad, p = make_test_ad_p()
#     iad_out = p.skyCorrectFromSlit([iad], function="spline3", order=5,
#                                    grow=2, niter=3, lsigma=3, hsigma=3,
#                                    aperture_growth=2, interactive=True)[0]
#
#     assert(ad_compare(ad_out, iad_out))
#
#
# @interactive_test
# def test_sky_correct_from_slit_with_multiple_sources_interactive():
#     width = 200
#     height = 100
#     n_sky_lines = 50
#     np.random.seed(0)
#
#     y0 = height // 2
#     y1 = 7 * height // 16
#     fwhm = 0.05 * height
#
#     source = (
#         fake_point_source_spatial_profile(height, width, {'c0': y0, 'c1': 0.0}, fwhm=fwhm) +
#         fake_point_source_spatial_profile(height, width, {'c0': y1, 'c1': 0.0}, fwhm=fwhm)
#     )
#
#     sky = SkyLines(n_sky_lines, width - 1)
#
#     def make_test_ad_p():
#         ad = create_zero_filled_fake_astrodata(height, width)
#
#         ad[0].data += source
#         ad[0].data += sky(ad[0].data, axis=1)
#         ad[0].APERTURE = get_aperture_table(height, width, center=height // 2)
#         # Ensure a new row is added correctly, regardless of column order
#         new_row = {'number': 2, 'c0': y1, 'aper_lower': -3, 'aper_upper': 3}
#         ad[0].APERTURE.add_row([new_row[c] for c in ad[0].APERTURE.colnames])
#
#         return ad, primitives_spect.Spect([])
#
#     ad, p = make_test_ad_p()
#     ad_out = p.skyCorrectFromSlit([ad], function="spline3", order=5,
#                                   grow=2, niter=3, lsigma=3, hsigma=3,
#                                   aperture_growth=2, interactive=False)[0]
#
#     iad, p = make_test_ad_p()
#     iad_out = p.skyCorrectFromSlit([iad], function="spline3", order=5,
#                                    grow=2, niter=3, lsigma=3, hsigma=3,
#                                    aperture_growth=2, interactive=True)[0]
#
#     assert(ad_compare(ad_out, iad_out))
