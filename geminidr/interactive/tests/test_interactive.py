from geminidr.core import primitives_spect
from geminidr.core.tests.test_spect import create_zero_filled_fake_astrodata, fake_point_source_spatial_profile, \
    get_aperture_table
from geminidr.interactive import interactive
import gemini_instruments


def interactive_test(tst):
    def set_test_mode():
        save_itm = interactive.test_mode
        interactive.test_mode = True
        tst()
        interactive.test_mode = save_itm
    return set_test_mode


@interactive_test
def test_trace_apertures_interactive():
    # Input parameters ----------------
    width = 400
    height = 200
    trace_model_parameters = {'c0': height // 2, 'c1': 5.0, 'c2': -0.5, 'c3': 0.5}

    # Boilerplate code ----------------
    # def make_test_ad():
    #     ad = create_zero_filled_fake_astrodata(height, width)
    #     ad[0].data += fake_point_source_spatial_profile(height, width, trace_model_parameters)
    #     ad[0].APERTURE = get_aperture_table(height, width)
    #     return ad

    # Running the test ----------------
    ad = create_zero_filled_fake_astrodata(height, width)
    ad[0].data += fake_point_source_spatial_profile(height, width, trace_model_parameters)
    ad[0].APERTURE = get_aperture_table(height, width)
    _p = primitives_spect.Spect([])
    ad_out = _p.traceApertures([ad], order=len(trace_model_parameters) + 1)[0]

    # # iad = make_test_ad()
    # iad = create_zero_filled_fake_astrodata(height, width)
    # iad[0].data += fake_point_source_spatial_profile(height, width, trace_model_parameters)
    # iad[0].APERTURE = get_aperture_table(height, width)
    # _p = primitives_spect.Spect([])
    # iad_out = _p.traceApertures([iad], order=len(trace_model_parameters) + 1, interactive=True)[0]

    # assert(ad_out == iad_out)
    pass

