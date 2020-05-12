import astrodata
import gemini_instruments

from astropy.io import fits
from gempy.gemini import gemini_tools as gt


PKG = 'geminidr.f2.lookups'


# Support functions specific to these tests:
def dummy_ad():
    """Create a dummy single-extension AD object"""
    phu = fits.PrimaryHDU(data=None)
    phu.header['INSTRUME'] = 'F2'
    phu.header['OBSERVAT'] = 'Gemini-South'
    phu.header['TELESCOP'] = 'Gemini-South'
    phu.header['MOSPOS'] = 'Open'
    phu.header['PIXSCALE'] = 0.18
    phu.header['GPREPARE'] = 'yes'
    phu.header['GRISM'] = 'Open'
    sci = fits.ImageHDU(data=[])
    sci.header['EXTNAME'] = 'SCI'
    sci.header['EXTVER'] = 1
    hdulist = fits.HDUList([phu, sci])
    return astrodata.open(hdulist)


def get_ad_sublist(adlist, names):
    """
    Select a sublist of AstroData instances from the input list using a list
    of filename strings. Any filenames that don't exist in the AstroData list
    are just ignored.
    """
    outlist = []
    for ad in adlist:
        if ad.filename in names:
            outlist.append(ad)
    return outlist


def same_lists(first, second):
    """Do the 2 input lists contain equivalent objects?"""
    nobjs = len(first)
    if len(second) != nobjs:
        return False

    # This is a bit of a brute-force approach since I'm not sure whether
    # ExposureGroups will get ordered consistently by a sort operation if
    # their internal ordering varies without defining the appropriate
    # special methods, but no big deal. Equivalence *does* work where the
    # internal ordering of ExposureGroups varies.
    for obj1 in first:
        found = False
        for obj2 in second:
            if obj1 == obj2:
                found = True
                break
        if not found:
            return False
    return True


def offdict_to_adlist(offdict):
    adlist = []
    for item in offdict:
        a = dummy_ad()
        a.phu['POFFSET'] = offdict[item][0]
        a.phu['QOFFSET'] = offdict[item][1]
        a.filename = item
        adlist.append(a)
    return adlist


def test_a_dith():
    # These are from Sandy's Q-15 programme:
    patt = {'S20140104S0094': (0., -0.),
            'S20140104S0095': (15., 15.),
            'S20140104S0096': (-5.58580959265E-15, 15.),
            'S20140104S0097': (-15., 15.),
            'S20140104S0098': (-15., -5.58580959265E-15),
            'S20140104S0099': (15., 5.58580959265E-15),
            'S20140104S0100': (15., -15.),
            'S20140104S0101': (5.58580959265E-15, -15.)}
    obj = ['S20140104S0094', 'S20140104S0095', 'S20140104S0096',
           'S20140104S0097', 'S20140104S0098', 'S20140104S0099',
           'S20140104S0100', 'S20140104S0101']

    adpatt = offdict_to_adlist(patt)
    objgroup = gt.ExposureGroup(get_ad_sublist(adpatt, obj))

    assert same_lists(gt.group_exposures(adpatt, pkg=PKG), (objgroup,))

def test_abba():
    # Simple 7' nod to sky without dithering (probably overly simple):
    patt = {'test01': (0., 0.),
            'test02': (210., 363.734),
            'test03': (210., 363.734),
            'test04': (0., 0.),
            'test05': (0., 0.),
            'test06': (210., 363.734),
            'test07': (210., 363.734),
            'test08': (0., 0.)}

    obj = ['test01', 'test04', 'test05', 'test08']
    sky = ['test02', 'test03', 'test06', 'test07']

    adpatt = offdict_to_adlist(patt)
    objgroup = gt.ExposureGroup(get_ad_sublist(adpatt, obj))
    skygroup = gt.ExposureGroup(get_ad_sublist(adpatt, sky))

    assert same_lists(gt.group_exposures(adpatt, pkg=PKG), (objgroup, skygroup))

def test_abbaacca():
    # Simple 7' nod to sky in opposite directions:
    patt = {'test01': (0., 0.),
            'test02': (210., 363.734),
            'test03': (210., 363.734),
            'test04': (0., 0.),
            'test05': (0., 0.),
            'test06': (-210., -363.734),
            'test07': (-210., -363.734),
            'test08': (0., 0.)}

    obj = ['test01', 'test04', 'test05', 'test08']
    sky1 = ['test02', 'test03']
    sky2 = ['test06', 'test07']

    adpatt = offdict_to_adlist(patt)
    objgroup = gt.ExposureGroup(get_ad_sublist(adpatt, obj))
    skygroup1 = gt.ExposureGroup(get_ad_sublist(adpatt, sky1))
    skygroup2 = gt.ExposureGroup(get_ad_sublist(adpatt, sky2))

    assert same_lists(gt.group_exposures(adpatt, pkg=PKG),
                      (objgroup, skygroup1, skygroup2))

def test_abba_dith_1():
    # Dither 2x2 on source and on sky, with ~50% overlap between A&B fields
    # (borderline case for grouping):
    patt = {'test01': (-5., -5.),
            'test02': (5., 5.),
            'test03': (-5., 175.),
            'test04': (5., 185.),
            'test05': (-5., 185.),
            'test06': (5., 175.),
            'test07': (5., -5.),
            'test08': (-5., 5.)}

    obj = ['test01', 'test02', 'test07', 'test08']
    sky = ['test03', 'test04', 'test05', 'test06']

    adpatt = offdict_to_adlist(patt)
    objgroup = gt.ExposureGroup(get_ad_sublist(adpatt, obj))
    skygroup = gt.ExposureGroup(get_ad_sublist(adpatt, sky))

    assert same_lists(gt.group_exposures(adpatt, pkg=PKG, frac_FOV=0.9),
                      (objgroup, skygroup))

def test_abba_dith_2():
    # Dither on source and on sky, from GS-F2-RECOM13-RUN-1-124:
    patt = {'S20130427S0199': (0., 0.),
            'S20130427S0200': (-20., -20.),
            'S20130427S0201': (10., -30.),
            'S20130427S0202': (0., 450.),
            'S20130427S0203': (-20., 430.),
            'S20130427S0204': (10., 420.)}

    obj = ['S20130427S0199', 'S20130427S0200', 'S20130427S0201']
    sky = ['S20130427S0202', 'S20130427S0203', 'S20130427S0204']

    adpatt = offdict_to_adlist(patt)
    objgroup = gt.ExposureGroup(get_ad_sublist(adpatt, obj))
    skygroup = gt.ExposureGroup(get_ad_sublist(adpatt, sky))

    assert same_lists(gt.group_exposures(adpatt, pkg=PKG), (objgroup, skygroup))

def test_abcde_dith():
    # A more exotic nod pattern between 3 sky & 2 off-centre object fields
    # with slight overlap between pointings. This particular pattern may not
    # be possible due to guide probe limits but it should still serve
    # to test the kind of thing PIs occasionally do with
    # smaller-field instruments, eg. using P2.

    patt = {'test01': (-315., 285.),
            'test02': (-285., 315.),
            'test03': (-166., -9.),
            'test04': (-136., 21.),
            'test05': (385., 285.),
            'test06': (415., 315.),
            'test07': (184., -33.),
            'test08': (214., -3.),
            'test09': (-15., -415.),
            'test10': (15., -385.)}

    sky1 = ['test01', 'test02']
    obj1 = ['test03', 'test04']
    sky2 = ['test05', 'test06']
    obj2 = ['test07', 'test08']
    sky3 = ['test09', 'test10']

    adpatt = offdict_to_adlist(patt)
    skygroup1 = gt.ExposureGroup(get_ad_sublist(adpatt, sky1))
    objgroup1 = gt.ExposureGroup(get_ad_sublist(adpatt, obj1))
    skygroup2 = gt.ExposureGroup(get_ad_sublist(adpatt, sky2))
    objgroup2 = gt.ExposureGroup(get_ad_sublist(adpatt, obj2))
    skygroup3 = gt.ExposureGroup(get_ad_sublist(adpatt, sky3))

    assert same_lists(gt.group_exposures(adpatt, pkg=PKG),
                      (skygroup1, objgroup1,
                       skygroup2, objgroup2, skygroup3))
