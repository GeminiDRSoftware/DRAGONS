from geminidr.core import primitives_spect
from geminidr.core.tests import ad_compare
from geminidr.core.tests.test_spect import create_zero_filled_fake_astrodata, fake_point_source_spatial_profile, \
    get_aperture_table
from geminidr.interactive import server
import gemini_instruments


def interactive_test(tst):
    def set_test_mode():
        save_itm = server.test_mode
        server.test_mode = True
        tst()
        server.test_mode = save_itm
    return set_test_mode


@interactive_test
def test_trace_apertures_interactive():
    print("starting test_trace_apertures_interactive")
    # Input parameters ----------------
    width = 400
    height = 200
    trace_model_parameters = {'c0': height // 2, 'c1': 5.0, 'c2': -0.5, 'c3': 0.5}

    # Boilerplate code ----------------
    def make_test_ad_p():
        ad = create_zero_filled_fake_astrodata(height, width)
        ad[0].data += fake_point_source_spatial_profile(height, width, trace_model_parameters)
        ad[0].APERTURE = get_aperture_table(height, width)
        return ad, primitives_spect.Spect([])

    # Running the test ----------------
    ad, _p = make_test_ad_p()
    ad_out = _p.traceApertures([ad], order=len(trace_model_parameters) + 1)[0]

    iad, _p = make_test_ad_p()
    iad_out = _p.traceApertures([iad], order=len(trace_model_parameters) + 1, interactive=True)[0]

    assert(ad_compare(ad_out, iad_out))

