import pytest
from geminidr.core.primitives_calibdb import _update_datalab
from geminidr.gemini import primitives_gemini as gemini
import astrodata

# TODO there is a lot of duplication within the tests, so this could be refactored out
#  into fixtures.. but there are some tweaks such as for slitIllum and the resulting
#  tests would be much harder to follow.  For now, I made the decision towards clarity.


global datalab_idx
datalab_idx = 1


def make_ad(instr='GMOS-N', typ='IMAGE', idx=1):
    astrofaker = pytest.importorskip('astrofaker')
    ad = astrofaker.create('NIRI', 'IMAGE')
    kw_datalab = ad._keyword_for('data_label')
    orig_datalab = f'GN-2021A-Q-1-1-{idx:03}'
    ad.phu[kw_datalab] = orig_datalab
    return ad


@pytest.fixture()
def ad():
    global datalab_idx
    ad = make_ad(instr='NIRI', typ='IMAGE')
    datalab_idx = datalab_idx + 1
    return ad


def test_update_datalab(ad, idx=1):
    kw_lut = {'DATALAB': 'comment'}
    kw_datalab = ad._keyword_for('data_label')
    orig_datalab = f'GN-2001A-Q-9-52-{idx:03}'
    ad.phu[kw_datalab] = orig_datalab
    _update_datalab(ad, '_flat', kw_lut)
    assert ad.phu[kw_datalab] == orig_datalab + '-FLAT'
    _update_datalab(ad, '_flat', kw_lut)
    assert ad.phu[kw_datalab] == orig_datalab + '-FLAT'
    _update_datalab(ad, '_bias', kw_lut)
    assert ad.phu[kw_datalab] == orig_datalab + '-BIAS'


def test_set_calibration(ad, monkeypatch):
    """
    setCalibration just calls down to the internal caldb.

    This test verifies that the calldown happens

    :param ad: fake astrodata to use for the Gemini instance
    :param monkeypatch: for unit test mocking
    :return:
    """
    test_gemini = gemini.Gemini([ad])
    global test_adinputs
    global test_parms
    test_adinputs = None
    test_parms = None
    def mock_set_calibrations(adinputs, **parms):
        global test_adinputs
        global test_parms
        test_adinputs = adinputs
        test_parms = parms
    monkeypatch.setattr(test_gemini.caldb, 'set_calibrations', mock_set_calibrations)
    adinputs = [make_ad()]  # we are just going to check these passed through, so it doesn't have to be real data
    test_gemini.setCalibration(adinputs, caltype='processed_arc', calfile="foo.fits")

    # check that Gemini properly passed down the parameters to the caldb
    assert(test_adinputs == adinputs)
    assert(test_parms == {'caltype': 'processed_arc', 'calfile': 'foo.fits'})


def test_get_processed_arc(ad, monkeypatch):
    adinputs = [ad]
    test_gemini = gemini.Gemini(adinputs)
    global saved_adinputs
    global saved_procmode
    saved_adinputs = None
    saved_procmode = None
    def mock_get_processed_arc(adinputs, procmode=None):
        global saved_adinputs
        global saved_procmode
        saved_adinputs = adinputs
        saved_procmode = procmode
        return {}
    monkeypatch.setattr(test_gemini.caldb, "get_processed_arc", mock_get_processed_arc)
    test_gemini.mode = 'ql'
    adinputs_out = test_gemini.getProcessedArc(adinputs)
    assert(saved_adinputs == adinputs)
    assert(saved_procmode == None)
    assert(adinputs_out == adinputs)
    test_gemini.mode = 'sq'
    adinputs_out = test_gemini.getProcessedArc(adinputs)
    assert(saved_adinputs == adinputs)
    assert(saved_procmode == 'sq')
    assert(adinputs_out == adinputs)


def test_get_processed_bias(ad, monkeypatch):
    adinputs = [ad]
    test_gemini = gemini.Gemini(adinputs)
    global saved_adinputs
    global saved_procmode
    saved_adinputs = None
    saved_procmode = None
    def mock_get_processed_bias(adinputs, procmode=None):
        global saved_adinputs
        global saved_procmode
        saved_adinputs = adinputs
        saved_procmode = procmode
        return {}
    monkeypatch.setattr(test_gemini.caldb, "get_processed_bias", mock_get_processed_bias)
    test_gemini.mode = 'ql'
    adinputs_out = test_gemini.getProcessedBias(adinputs)
    assert(saved_adinputs == adinputs)
    assert(saved_procmode == None)
    assert(adinputs_out == adinputs)
    test_gemini.mode = 'sq'
    adinputs_out = test_gemini.getProcessedBias(adinputs)
    assert(saved_adinputs == adinputs)
    assert(saved_procmode == 'sq')
    assert(adinputs_out == adinputs)


def test_get_processed_dark(ad, monkeypatch):
    adinputs = [ad]
    test_gemini = gemini.Gemini(adinputs)
    global saved_adinputs
    global saved_procmode
    saved_adinputs = None
    saved_procmode = None
    def mock_get_processed_dark(adinputs, procmode=None):
        global saved_adinputs
        global saved_procmode
        saved_adinputs = adinputs
        saved_procmode = procmode
        return {}
    monkeypatch.setattr(test_gemini.caldb, "get_processed_dark", mock_get_processed_dark)
    test_gemini.mode = 'ql'
    adinputs_out = test_gemini.getProcessedDark(adinputs)
    assert(saved_adinputs == adinputs)
    assert(saved_procmode == None)
    assert(adinputs_out == adinputs)
    test_gemini.mode = 'sq'
    adinputs_out = test_gemini.getProcessedDark(adinputs)
    assert(saved_adinputs == adinputs)
    assert(saved_procmode == 'sq')
    assert(adinputs_out == adinputs)


def test_get_processed_flat(ad, monkeypatch):
    adinputs = [ad]
    test_gemini = gemini.Gemini(adinputs)
    global saved_adinputs
    global saved_procmode
    saved_adinputs = None
    saved_procmode = None
    def mock_get_processed_flat(adinputs, procmode=None):
        global saved_adinputs
        global saved_procmode
        saved_adinputs = adinputs
        saved_procmode = procmode
        return {}
    monkeypatch.setattr(test_gemini.caldb, "get_processed_flat", mock_get_processed_flat)
    test_gemini.mode = 'ql'
    adinputs_out = test_gemini.getProcessedFlat(adinputs)
    assert(saved_adinputs == adinputs)
    assert(saved_procmode == None)
    assert(adinputs_out == adinputs)
    test_gemini.mode = 'sq'
    adinputs_out = test_gemini.getProcessedFlat(adinputs)
    assert(saved_adinputs == adinputs)
    assert(saved_procmode == 'sq')
    assert(adinputs_out == adinputs)


def test_get_processed_fringe(ad, monkeypatch):
    adinputs = [ad]
    test_gemini = gemini.Gemini(adinputs)
    global saved_adinputs
    global saved_procmode
    saved_adinputs = None
    saved_procmode = None
    def mock_get_processed_fringe(adinputs, procmode=None):
        global saved_adinputs
        global saved_procmode
        saved_adinputs = adinputs
        saved_procmode = procmode
        return {}
    monkeypatch.setattr(test_gemini.caldb, "get_processed_fringe", mock_get_processed_fringe)
    test_gemini.mode = 'ql'
    adinputs_out = test_gemini.getProcessedFringe(adinputs)
    assert(saved_adinputs == adinputs)
    assert(saved_procmode == None)
    assert(adinputs_out == adinputs)
    test_gemini.mode = 'sq'
    adinputs_out = test_gemini.getProcessedFringe(adinputs)
    assert(saved_adinputs == adinputs)
    assert(saved_procmode == 'sq')
    assert(adinputs_out == adinputs)


def test_get_processed_standard(ad, monkeypatch):
    adinputs = [ad]
    test_gemini = gemini.Gemini(adinputs)
    global saved_adinputs
    global saved_procmode
    saved_adinputs = None
    saved_procmode = None
    def mock_get_processed_standard(adinputs, procmode=None):
        global saved_adinputs
        global saved_procmode
        saved_adinputs = adinputs
        saved_procmode = procmode
        return {}
    monkeypatch.setattr(test_gemini.caldb, "get_processed_standard", mock_get_processed_standard)
    test_gemini.mode = 'ql'
    adinputs_out = test_gemini.getProcessedStandard(adinputs)
    assert(saved_adinputs == adinputs)
    assert(saved_procmode == None)
    assert(adinputs_out == adinputs)
    test_gemini.mode = 'sq'
    adinputs_out = test_gemini.getProcessedStandard(adinputs)
    assert(saved_adinputs == adinputs)
    assert(saved_procmode == 'sq')
    assert(adinputs_out == adinputs)


def test_get_processed_slitillum(ad, monkeypatch):
    adinputs = [ad]
    test_gemini = gemini.Gemini(adinputs)
    global saved_adinputs
    global saved_procmode
    saved_adinputs = None
    saved_procmode = None
    def mock_get_processed_slitillum(adinputs, procmode=None):
        global saved_adinputs
        global saved_procmode
        saved_adinputs = adinputs
        saved_procmode = procmode
        return {}
    monkeypatch.setattr(test_gemini.caldb, "get_processed_slitillum", mock_get_processed_slitillum)
    test_gemini.mode = 'ql'
    adinputs_out = test_gemini.getProcessedSlitIllum(adinputs)
    assert(saved_adinputs == adinputs)
    assert(saved_procmode == None)
    assert(adinputs_out == adinputs)
    test_gemini.mode = 'sq'
    adinputs_out = test_gemini.getProcessedSlitIllum(adinputs)
    assert(saved_adinputs == adinputs)
    assert(saved_procmode == 'sq')
    assert(adinputs_out == adinputs)


def test_get_mdf(ad, monkeypatch):
    adinputs = [ad]
    test_gemini = gemini.Gemini(adinputs)
    global saved_adinputs
    global saved_caltype
    saved_adinputs = None
    saved_caltype = None
    def mock_get_calibrations(adinputs, caltype=None):
        global saved_adinputs
        global saved_caltype
        saved_adinputs = adinputs
        saved_caltype = caltype
        return {}
    monkeypatch.setattr(test_gemini.caldb, "get_calibrations", mock_get_calibrations)
    adinputs_out = test_gemini.getMDF(adinputs)
    assert(saved_adinputs == adinputs)
    assert(saved_caltype == "mask")
    assert(adinputs_out == adinputs)


def test_store_calibration(ad, monkeypatch):
    cals1 = [make_ad('GMOS-N', 'IMAGE'), make_ad('GMOS-N', 'IMAGE')]
    cals2 = [make_ad('GMOS-N', 'IMAGE'), make_ad('GMOS-N', 'IMAGE')]
    test_gemini = gemini.Gemini([ad])
    storecal_args = list()
    def mock_store_calibration(cals, caltype):
        storecal_args.append((cals, caltype))
    monkeypatch.setattr(test_gemini, "storeCalibration", mock_store_calibration)
    test_gemini.storeCalibration(cals1, 'bias')
    test_gemini.storeCalibration(cals2, 'flat')
    assert(len(storecal_args) == 2)
    assert(storecal_args[0][0] == cals1)
    assert(storecal_args[0][1] == 'bias')
    assert(storecal_args[1][0] == cals2)
    assert(storecal_args[1][1] == 'flat')


def test_store_calibration(ad, monkeypatch):
    cals1 = [make_ad('GMOS-N', 'IMAGE'), make_ad('GMOS-N', 'IMAGE')]
    cals2 = [make_ad('GMOS-N', 'IMAGE'), make_ad('GMOS-N', 'IMAGE')]
    test_gemini = gemini.Gemini([ad])
    storecal_args = list()
    def mock_store_calibration(cals, caltype):
        storecal_args.append((cals, caltype))
    monkeypatch.setattr(test_gemini, "storeCalibration", mock_store_calibration)
    test_gemini.storeCalibration(cals1, 'bias')
    test_gemini.storeCalibration(cals2, 'flat')
    assert(len(storecal_args) == 2)
    assert(storecal_args[0][0] == cals1)
    assert(storecal_args[0][1] == 'bias')
    assert(storecal_args[1][0] == cals2)
    assert(storecal_args[1][1] == 'flat')


def test_store_processed_arc(ad, monkeypatch):
    arc = make_ad(instr='GMOS-N', typ='IMAGE', idx=1)
    test_gemini = gemini.Gemini([ad])
    test_gemini.mode = 'ql'
    storecal_args = list()
    def mock_store_calibration(cals, caltype):
        storecal_args.append((cals, caltype))
    monkeypatch.setattr(test_gemini, "storeCalibration", mock_store_calibration)
    test_gemini.storeProcessedArc([arc], force=False)
    assert(len(storecal_args) == 1)
    assert(len(storecal_args[0][0]) == 1)
    assert(storecal_args[0][1] == 'processed_arc')
    saved_arc = storecal_args[0][0][0]
    assert(saved_arc.filename == 'N20010101S0001_ql_arc.fits')
    assert(saved_arc.phu['PROCMODE'] == 'ql')
    assert(saved_arc.phu['PROCARC'])
    assert(saved_arc.phu['DATALAB'] == 'GN-2021A-Q-1-1-001-ARC')


def test_store_processed_bias(ad, monkeypatch):
    bias = make_ad(instr='GMOS-N', typ='IMAGE', idx=1)
    test_gemini = gemini.Gemini([ad])
    test_gemini.mode = 'ql'
    storecal_args = list()
    def mock_store_calibration(cals, caltype):
        storecal_args.append((cals, caltype))
    monkeypatch.setattr(test_gemini, "storeCalibration", mock_store_calibration)
    test_gemini.storeProcessedBias([bias], force=False)
    assert(len(storecal_args) == 1)
    assert(len(storecal_args[0][0]) == 1)
    assert(storecal_args[0][1] == 'processed_bias')
    saved_arc = storecal_args[0][0][0]
    assert(saved_arc.filename == 'N20010101S0001_ql_bias.fits')
    assert(saved_arc.phu['PROCMODE'] == 'ql')
    assert(saved_arc.phu['PROCBIAS'])
    assert(saved_arc.phu['DATALAB'] == 'GN-2021A-Q-1-1-001-BIAS')


def test_store_processed_dark(ad, monkeypatch):
    dark = make_ad(instr='GMOS-N', typ='IMAGE', idx=1)
    test_gemini = gemini.Gemini([ad])
    test_gemini.mode = 'ql'
    storecal_args = list()
    def mock_store_calibration(cals, caltype):
        storecal_args.append((cals, caltype))
    monkeypatch.setattr(test_gemini, "storeCalibration", mock_store_calibration)
    test_gemini.storeProcessedDark([dark], force=False)
    assert(len(storecal_args) == 1)
    assert(len(storecal_args[0][0]) == 1)
    assert(storecal_args[0][1] == 'processed_dark')
    saved_arc = storecal_args[0][0][0]
    assert(saved_arc.filename == 'N20010101S0001_ql_dark.fits')
    assert(saved_arc.phu['PROCMODE'] == 'ql')
    assert(saved_arc.phu['PROCDARK'])
    assert(saved_arc.phu['DATALAB'] == 'GN-2021A-Q-1-1-001-DARK')


def test_store_processed_flat(ad, monkeypatch):
    flat = make_ad(instr='GMOS-N', typ='IMAGE', idx=1)
    test_gemini = gemini.Gemini([ad])
    test_gemini.mode = 'ql'
    storecal_args = list()
    def mock_store_calibration(cals, caltype):
        storecal_args.append((cals, caltype))
    monkeypatch.setattr(test_gemini, "storeCalibration", mock_store_calibration)
    test_gemini.storeProcessedFlat([flat], force=False)
    assert(len(storecal_args) == 1)
    assert(len(storecal_args[0][0]) == 1)
    assert(storecal_args[0][1] == 'processed_flat')
    saved_arc = storecal_args[0][0][0]
    assert(saved_arc.filename == 'N20010101S0001_ql_flat.fits')
    assert(saved_arc.phu['PROCMODE'] == 'ql')
    assert(saved_arc.phu['PROCFLAT'])
    assert(saved_arc.phu['DATALAB'] == 'GN-2021A-Q-1-1-001-FLAT')


def test_store_processed_fringe(ad, monkeypatch):
    frng = make_ad(instr='GMOS-N', typ='IMAGE', idx=1)
    test_gemini = gemini.Gemini([ad])
    test_gemini.mode = 'ql'
    storecal_args = list()
    def mock_store_calibration(cals, caltype):
        storecal_args.append((cals, caltype))
    monkeypatch.setattr(test_gemini, "storeCalibration", mock_store_calibration)
    test_gemini.storeProcessedFringe([frng])
    assert(len(storecal_args) == 1)
    assert(len(storecal_args[0][0]) == 1)
    assert(storecal_args[0][1] == 'processed_fringe')
    saved_arc = storecal_args[0][0][0]
    assert(saved_arc.filename == 'N20010101S0001_ql_fringe.fits')
    assert(saved_arc.phu['PROCMODE'] == 'ql')
    assert(saved_arc.phu['PROCFRNG'])
    # note that storeProcessedFringe explicitly does not update the datalabel
    assert(saved_arc.phu['DATALAB'] == 'GN-2021A-Q-1-1-001')


def test_store_processed_science(ad, monkeypatch):
    sci = make_ad(instr='GMOS-N', typ='IMAGE', idx=1)
    test_gemini = gemini.Gemini([ad])
    test_gemini.mode = 'ql'
    storecal_args = list()
    def mock_store_calibration(cals, caltype):
        storecal_args.append((cals, caltype))
    monkeypatch.setattr(test_gemini, "storeCalibration", mock_store_calibration)
    # make a plausible processed science filename
    sci.filename = ad.filename[:-5] + '_stack.fits'

    test_gemini.storeProcessedScience([sci])
    assert(len(storecal_args) == 0)  # if not upload, not called
    assert(sci.filename == 'N20010101S0001_stack.fits')
    assert(sci.phu['PROCMODE'] == 'ql')
    assert(sci.phu['PROCSCI'])
    assert(sci.phu['DATALAB'] == 'GN-2021A-Q-1-1-001')


def test_store_processed_standard(ad, monkeypatch):
    std = make_ad(instr='GMOS-N', typ='IMAGE', idx=1)
    test_gemini = gemini.Gemini([ad])
    test_gemini.mode = 'ql'
    storecal_args = list()
    def mock_store_calibration(cals, caltype):
        storecal_args.append((cals, caltype))
    monkeypatch.setattr(test_gemini, "storeCalibration", mock_store_calibration)
    test_gemini.storeProcessedStandard([std])
    assert(len(storecal_args) == 1)
    assert(len(storecal_args[0][0]) == 1)
    assert(storecal_args[0][1] == 'processed_standard')
    saved_arc = storecal_args[0][0][0]
    assert(saved_arc.filename == 'N20010101S0001_ql_standard.fits')
    assert(saved_arc.phu['PROCMODE'] == 'ql')
    assert(saved_arc.phu['PROCSTND'])
    assert(saved_arc.phu['DATALAB'] == 'GN-2021A-Q-1-1-001-STANDARD')


def test_store_processed_slitillum(ad, monkeypatch):
    illum = make_ad(instr='GMOS-N', typ='IMAGE', idx=1)
    illum.phu['MAKESILL'] = '1'
    test_gemini = gemini.Gemini([ad])
    test_gemini.mode = 'ql'
    storecal_args = list()
    def mock_store_calibration(cals, caltype):
        storecal_args.append((cals, caltype))
    monkeypatch.setattr(test_gemini, "storeCalibration", mock_store_calibration)
    test_gemini.storeProcessedSlitIllum([illum])
    assert(len(storecal_args) == 1)
    assert(len(storecal_args[0][0]) == 1)
    assert(storecal_args[0][1] == 'processed_slitillum')
    saved_arc = storecal_args[0][0][0]
    assert(saved_arc.filename == 'N20010101S0001_ql_slitIllum.fits')
    assert(saved_arc.phu['PROCMODE'] == 'ql')
    assert(saved_arc.phu['PROCILLM'])
    assert(saved_arc.phu['DATALAB'] == 'GN-2021A-Q-1-1-001-SLITILLUM')


def test_store_bpm(ad, monkeypatch):
    bpm = make_ad(instr='GMOS-N', typ='IMAGE', idx=1)
    test_gemini = gemini.Gemini([ad])
    test_gemini.mode = 'ql'
    storecal_args = list()
    def mock_store_calibration(cals, caltype):
        storecal_args.append((cals, caltype))
    monkeypatch.setattr(test_gemini, "storeCalibration", mock_store_calibration)
    test_gemini.storeBPM([bpm])
    assert(len(storecal_args) == 1)
    assert(len(storecal_args[0][0]) == 1)
    assert(storecal_args[0][1] == 'bpm')
    saved_arc = storecal_args[0][0][0]
    assert(saved_arc.filename == 'N20010101S0001_ql_bpm.fits')
    assert(saved_arc.phu['PROCMODE'] == 'ql')
    assert(saved_arc.phu['BPM'])
    # not checking datalab, it will be quite different from input, unlike the various cal types
