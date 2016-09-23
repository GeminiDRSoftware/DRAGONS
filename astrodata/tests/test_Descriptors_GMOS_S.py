#
#                                                                  gemini_python
#
#                                                     test_Descriptors_GMOS_S.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
""" Descriptor tests on GMOS_S Image data type.

    To run theses tests w/ pytest:

    $ py.test [-v] -rxX

      -v is verbose. See pytest help for more options.
"""
import os
import sys
import pytest

from astrodata import AstroData
# ==================================== Set up  =================================
# TESTFILEs are located under gemini_python/test_data/astrodata_bench/
GEM0 = None
GEM1 = None
GEM0 = os.environ.get("ADTESTDATA")
# If no env var, look for 'test_data' under gemini_python
if GEM0 is None:
    for path in sys.path:
        if 'gemini_python' in path:
            GEM1 = os.path.join(path.split('gemini_python')[0], 
                               'gemini_python', 'test_data')
            break

if GEM0:
    TESTFILE  = os.path.join(GEM0, 'GS_GMOS_IMAGE.fits') # 3 'SCI'
elif GEM1:
    TESTFILE  = os.path.join(GEM1, 'astrodata_bench/GS_GMOS_IMAGE.fits') # 3 'SCI'
else:
    SystemExit("Cannot find astrodata benchmark test_data in available paths")
# ==================================== tests  ==================================
xfail = pytest.mark.xfail
# ==============================================================================
def test_airmass():
    ad = AstroData(TESTFILE)
    assert 1.312 == ad.airmass().get_value()

def test_amp_read_area():
    ad = AstroData(TESTFILE)
    assert len(ad.amp_read_area().as_list()) == 3

def test_array_name():
    ad = AstroData(TESTFILE)
    bench_array = ['EEV 2037-06-03', 'EEV 8194-19-04', 'EEV 8261-07-04']
    assert ad.array_name().as_list() == bench_array

def test_array_section():
    ad = AstroData(TESTFILE)
    bench_section = [0, 2048, 0, 4608]
    assert ad.array_section().as_list() == bench_section

def test_azimuth():
    ad = AstroData(TESTFILE)
    bench_az = 285.818
    assert round(ad.azimuth().get_value(), 3) == bench_az

def test_camera():
    ad = AstroData(TESTFILE)
    cam = 'GMOS-S'
    assert ad.camera().get_value() == cam

def test_cass_rotator_pa():
    rot_pa = 58.783
    ad = AstroData(TESTFILE)
    assert round(ad.cass_rotator_pa().get_value(), 3) == rot_pa

def test_central_wavelength():
    ad = AstroData(TESTFILE)
    clambda = 0.0
    assert ad.central_wavelength().get_value() == clambda

def test_coadds():
    ad = AstroData(TESTFILE)
    coadds = 1
    assert ad.coadds().get_value() == coadds

def test_data_label():
    ad = AstroData(TESTFILE)
    dl = 'GS-2013B-Q-32-63-001'
    assert ad.data_label().get_value() == dl

def test_data_section():
    ad = AstroData(TESTFILE)
    dsection = [[0, 1024, 0, 2304], [0, 1024, 0, 2304], 
                [32, 1056, 0, 2304]]
    assert ad.data_section().as_list() == dsection

def test_dec():
    ad = AstroData(TESTFILE)
    bench_dec = -13.436
    assert round(ad.dec().get_value(), 3) == bench_dec

def test_decker():
    ad = AstroData(TESTFILE)
    assert ad.decker().get_value() is None

def test_detector_name():
    ad = AstroData(TESTFILE)
    dname = 'EEV2037-06-03EEV8194-19-04EEV8261-07-04'
    assert ad.detector_name().get_value() == dname

def test_detector_roi_setting():
    ad = AstroData(TESTFILE)
    roi = 'Full Frame'
    assert ad.detector_roi_setting().get_value() == roi

def test_detector_rois_requested():
    ad = AstroData(TESTFILE)
    roi_req = [[1, 6144, 1, 4608]]
    assert ad.detector_rois_requested().get_value() == roi_req

def test_detector_section():
    ad = AstroData(TESTFILE)
    detsec = [[0, 2048, 0, 4608], 
              [2048, 4096, 0, 4608], 
              [4096, 6144, 0, 4608]]
    assert ad.detector_section().as_list() == detsec

def test_detector_x_bin():
    ad = AstroData(TESTFILE)
    xbin = 2
    assert ad.detector_x_bin().get_value() == xbin

def test_detector_y_bin():
    ad = AstroData(TESTFILE)
    ybin = 2
    assert ad.detector_y_bin().get_value() == ybin

def test_disperser():
    ad = AstroData(TESTFILE)
    disperser = 'MIRROR'
    assert ad.disperser().get_value() == disperser

def test_dispersion_1():
    ad = AstroData(TESTFILE)
    assert len(ad.dispersion().as_list()) == 3
    
def test_dispersion_2():
    ad = AstroData(TESTFILE)
    disp = [round(-4.04851184158353e-05, 3),
            round(-4.05918605309534e-05, 3),
            round(-4.04838168872906e-05, 3)]
    dvlist = ad.dispersion().as_list()
    assert round(dvlist[0], 3) == disp[0]
    assert round(dvlist[1], 3) == disp[1]
    assert round(dvlist[2], 3) == disp[2]

def test_dispersion_axis():
    ad = AstroData(TESTFILE)
    assert ad.dispersion_axis().get_value() == 1

def test_elevation():
    ad = AstroData(TESTFILE)
    el = round(49.7580375)
    assert round(ad.elevation().get_value()) == el

def test_exposure_time():
    ad = AstroData(TESTFILE)
    exp = round(30.4991180896759)
    assert round(ad.exposure_time().get_value()) == exp

def test_filter_name():
    ad = AstroData(TESTFILE)
    filtname = ('open1-6&i_G0327').lower()
    assert ad.filter_name().get_value().lower() == filtname

def test_focal_plane_mask():
    ad = AstroData(TESTFILE)
    fmask = 'Imaging'
    assert ad.focal_plane_mask().get_value() == fmask

def test_gain():
    ad = AstroData(TESTFILE)
    gains = [2.372, 2.076, 2.097]
    assert ad.gain().as_list() == gains

def test_gain_setting():
    ad = AstroData(TESTFILE)
    assert ad.gain_setting().get_value().lower() == 'low'

def test_grating():
    ad = AstroData(TESTFILE)
    grate = 'MIRROR'
    assert ad.grating().get_value() == grate

def test_group_id():
    ad = AstroData(TESTFILE)
    assert type(ad.group_id().as_list()) is list

def test_local_time():
    ad = AstroData(TESTFILE)
    isotime = '05:36:26.900000'
    assert ad.local_time().as_pytype().isoformat() == isotime

def test_lyot_stop():
    ad = AstroData(TESTFILE)
    assert ad.lyot_stop().get_value() is None

def test_mdf_row_id():
    ad = AstroData(TESTFILE)
    assert ad.mdf_row_id().get_value() is None

def test_nod_count():
    ad = AstroData(TESTFILE)
    assert ad.nod_count().get_value() is None

def test_nod_pixels():
    ad = AstroData(TESTFILE)
    assert ad.nod_pixels().get_value() is None

def test_nominal_atmospheric_extinction():
    ad = AstroData(TESTFILE)
    ext = round(0.024960000000000006, 5)
    assert round(ad.nominal_atmospheric_extinction().get_value(), 5) == ext

def test_nominal_photometric_zeropoint():
    ad = AstroData(TESTFILE)
    zp = [27.844, 27.849, 27.883]
    assert ad.nominal_photometric_zeropoint().as_list() == zp

def test_non_linear_level():
    ad = AstroData(TESTFILE)
    assert ad.non_linear_level().get_value() == 65535

def test_observation_class():
    ad = AstroData(TESTFILE)
    oclass = 'science'
    assert ad.observation_class().get_value().lower() == oclass

def test_observation_epoch():
    ad = AstroData(TESTFILE)
    epoch = round(2013.773740324, 5)
    assert round(ad.observation_epoch().get_value(), 5) == epoch

def test_observation_id():
    ad = AstroData(TESTFILE)
    oid = 'GS-2013B-Q-32-63'
    assert ad.observation_id().get_value() == oid

def test_pixel_scale():
    ad = AstroData(TESTFILE)
    pscale = 0.146
    assert ad.pixel_scale().get_value() == pscale

def test_prism():
    ad = AstroData(TESTFILE)
    assert ad.prism().get_value() is None

def test_program_id():
    ad = AstroData(TESTFILE)
    pid = 'GS-2013B-Q-32'
    assert ad.program_id().get_value() == pid

def test_pupil_mask():
    ad = AstroData(TESTFILE)
    assert ad.pupil_mask().get_value() is None

def test_qa_state():
    ad = AstroData(TESTFILE)
    assert ad.qa_state().get_value() == 'Pass'

def test_ra():
    ad = AstroData(TESTFILE)
    ra_val = round(37.53291667, 5)
    assert round(ad.ra().get_value(), 5) == ra_val

def test_raw_bg():
    ad = AstroData(TESTFILE)
    assert ad.raw_bg().get_value() is None

def test_raw_cc():
    ad = AstroData(TESTFILE)
    assert ad.raw_cc().get_value() is None

def test_raw_iq():
    ad = AstroData(TESTFILE)
    assert ad.raw_iq().get_value() is None

def test_raw_wv():
    ad = AstroData(TESTFILE)
    assert ad.raw_wv().get_value() is None

def test_read_mode():
    ad = AstroData(TESTFILE)
    assert ad.read_mode().get_value() == 'Normal'

def test_read_noise():
    ad = AstroData(TESTFILE)
    assert ad.read_noise().get_value() is None

def test_read_speed_setting():
    ad = AstroData(TESTFILE)
    assert ad.read_speed_setting().get_value() == 'slow'

def test_requested_bg():
    ad = AstroData(TESTFILE)
    assert ad.requested_bg().get_value() == 80

def test_requested_cc():
    ad = AstroData(TESTFILE)
    assert ad.requested_cc().get_value() == 70

def test_requested_iq():
    ad = AstroData(TESTFILE)
    assert ad.requested_iq().get_value() == 70

def test_requested_wv():
    ad = AstroData(TESTFILE)
    assert ad.requested_wv().get_value() == 100

def test_saturation_level():
    ad = AstroData(TESTFILE)
    assert ad.saturation_level().get_value() == 65535

def test_slit():
    ad = AstroData(TESTFILE)
    assert ad.slit().get_value() is None

def test_ut_datetime():
    ad = AstroData(TESTFILE)
    ut_date = '2013-10-10T08:36:27.400000'
    assert ad.ut_datetime().get_value().isoformat() == ut_date

def test_ut_time():
    ad = AstroData(TESTFILE)
    uttime = '08:36:27.400000'
    assert ad.ut_time().get_value().isoformat() == uttime

def test_wavefront_sensor():
    ad = AstroData(TESTFILE)
    assert ad.wavefront_sensor().get_value() == 'OIWFS'

def test_wavelength_band():
    ad = AstroData(TESTFILE)
    assert ad.wavelength_band().get_value() == 'i'

def test_wavelength_reference_pixel():
    ad = AstroData(TESTFILE)
    assert len(ad.wavelength_reference_pixel().as_list()) == 3

def test_wavelength_reference_pixel():
    ad = AstroData(TESTFILE)
    bench_ref_pix = [round(1555.8645612709, 5),
                     round(512.068306249868, 5),
                     round(-501.319404266684, 5)]

    dv = ad.wavelength_reference_pixel().as_list()
    assert round(dv[0], 5) == bench_ref_pix[0]
    assert round(dv[1], 5) == bench_ref_pix[1]
    assert round(dv[2], 5) == bench_ref_pix[2]

def test_well_depth_setting():
    ad = AstroData(TESTFILE)
    assert ad.wavelength_reference_pixel().get_value() is None

def test_x_offset():
    ad = AstroData(TESTFILE)
    assert ad.x_offset().get_value() == 0.0

def test_y_offset():
    ad = AstroData(TESTFILE)
    assert ad.y_offset().get_value() == 0.0
