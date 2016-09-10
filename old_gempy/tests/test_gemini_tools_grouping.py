from gempy.gemini import gemini_tools as gt

from gempy.tests import utils as tu

# Support functions specific to these tests:

def offdict_to_adlist(offdict):

    adlist = []

    for item in offdict:
        
        a = tu.dummy_ad()
        a.phu.header['INSTRUME'] = 'F2'
        a.phu.header['OBSERVAT'] = 'Gemini-South'
        a.phu.header['TELESCOP'] = 'Gemini-South'
        a.phu.header['MOSPOS'] = 'Open'
        a.phu.header['POFFSET'] = offdict[item][0]
        a.phu.header['QOFFSET'] = offdict[item][1]
        a.filename = item
        a.refresh_types()
        adlist.append(a)

    return adlist

# Tests:

def test_A_dith():
    # These are from Sandy's Q-15 programme:
    patt = {'S20140104S0094': (0.,-0.),
            'S20140104S0095': (15.,15.),
            'S20140104S0096': (-5.58580959265E-15,15.),
            'S20140104S0097': (-15.,15.),
            'S20140104S0098': (-15.,-5.58580959265E-15),
            'S20140104S0099': (15.,5.58580959265E-15),
            'S20140104S0100': (15.,-15.),
            'S20140104S0101': (5.58580959265E-15,-15.)
    }
    obj = ['S20140104S0094', 'S20140104S0095', 'S20140104S0096',
           'S20140104S0097', 'S20140104S0098', 'S20140104S0099',
           'S20140104S0100', 'S20140104S0101']

    adpatt = offdict_to_adlist(patt)
    objgroup = gt.ExposureGroup(tu.get_ad_sublist(adpatt, obj))

    assert tu.same_lists(gt.group_exposures(adpatt), (objgroup,))

def test_ABBA():
    # Simple 7' nod to sky without dithering (probably overly simple):
    patt = {'test01': (0.,0.),
            'test02': (210.,363.734),
            'test03': (210.,363.734),
            'test04': (0.,0.),
            'test05': (0.,0.),
            'test06': (210.,363.734),
            'test07': (210.,363.734),
            'test08': (0.,0.),
    }

    obj = ['test01', 'test04', 'test05', 'test08']
    sky = ['test02', 'test03', 'test06', 'test07']

    adpatt = offdict_to_adlist(patt)
    objgroup = gt.ExposureGroup(tu.get_ad_sublist(adpatt, obj))
    skygroup = gt.ExposureGroup(tu.get_ad_sublist(adpatt, sky))

    assert tu.same_lists(gt.group_exposures(adpatt), (objgroup, skygroup))

def test_ABBAACCA():
    # Simple 7' nod to sky in opposite directions:
    patt = {'test01': (0.,0.),
            'test02': (210.,363.734),
            'test03': (210.,363.734),
            'test04': (0.,0.),
            'test05': (0.,0.),
            'test06': (-210.,-363.734),
            'test07': (-210.,-363.734),
            'test08': (0.,0.),
    }

    obj = ['test01', 'test04', 'test05', 'test08']
    sky1 = ['test02', 'test03']
    sky2 = ['test06', 'test07']

    adpatt = offdict_to_adlist(patt)
    objgroup = gt.ExposureGroup(tu.get_ad_sublist(adpatt, obj))
    skygroup1 = gt.ExposureGroup(tu.get_ad_sublist(adpatt, sky1))
    skygroup2 = gt.ExposureGroup(tu.get_ad_sublist(adpatt, sky2))

    assert tu.same_lists(gt.group_exposures(adpatt), \
        (objgroup, skygroup1, skygroup2))

def test_ABBA_dith_1():
    # Dither 2x2 on source and on sky, with ~50% overlap between A&B fields
    # (borderline case for grouping):
    patt = {'test01': (-5.,-5.),
            'test02': (5.,5.),
            'test03': (-5.,175.),
            'test04': (5.,185.),
            'test05': (-5.,185.),
            'test06': (5.,175.),
            'test07': (5.,-5.),
            'test08': (-5.,5.)
    }

    obj = ['test01', 'test02', 'test07', 'test08']
    sky = ['test03', 'test04', 'test05', 'test06']

    adpatt = offdict_to_adlist(patt)
    objgroup = gt.ExposureGroup(tu.get_ad_sublist(adpatt, obj))
    skygroup = gt.ExposureGroup(tu.get_ad_sublist(adpatt, sky))

    assert tu.same_lists(gt.group_exposures(adpatt, frac_FOV=0.9), \
        (objgroup, skygroup))

def test_ABBA_dith_2():
    # Dither on source and on sky, from GS-F2-RECOM13-RUN-1-124:
    patt = {'S20130427S0199': (0.,0.),
            'S20130427S0200': (-20.,-20.),
            'S20130427S0201': (10.,-30.),
            'S20130427S0202': (0.,450.),
            'S20130427S0203': (-20.,430.),
            'S20130427S0204': (10.,420.),
    }

    obj = ['S20130427S0199', 'S20130427S0200', 'S20130427S0201']
    sky = ['S20130427S0202', 'S20130427S0203', 'S20130427S0204']

    adpatt = offdict_to_adlist(patt)
    objgroup = gt.ExposureGroup(tu.get_ad_sublist(adpatt, obj))
    skygroup = gt.ExposureGroup(tu.get_ad_sublist(adpatt, sky))

    assert tu.same_lists(gt.group_exposures(adpatt), (objgroup, skygroup))

def test_ABCDE_dith():
    # A more exotic nod pattern between 3 sky & 2 off-centre object fields
    # with slight overlap between pointings. This particular pattern may not
    # be possible due to guide probe limits but it should still serve to test
    # the kind of thing PIs occasionally do with smaller-field instruments,
    # eg. using P2.

    patt = {'test01': (-315.,285.),
            'test02': (-285.,315.),
            'test03': (-166.,-9.),
            'test04': (-136.,21.),
            'test05': (385.,285.),
            'test06': (415.,315.),
            'test07': (184.,-33.),
            'test08': (214.,-3.),
            'test09': (-15.,-415.),
            'test10': (15.,-385.)
    }

    sky1 = ['test01', 'test02']
    obj1 = ['test03', 'test04']
    sky2 = ['test05', 'test06']
    obj2 = ['test07', 'test08']
    sky3 = ['test09', 'test10']

    adpatt = offdict_to_adlist(patt)
    skygroup1 = gt.ExposureGroup(tu.get_ad_sublist(adpatt, sky1))
    objgroup1 = gt.ExposureGroup(tu.get_ad_sublist(adpatt, obj1))
    skygroup2 = gt.ExposureGroup(tu.get_ad_sublist(adpatt, sky2))
    objgroup2 = gt.ExposureGroup(tu.get_ad_sublist(adpatt, obj2))
    skygroup3 = gt.ExposureGroup(tu.get_ad_sublist(adpatt, sky3))

    assert tu.same_lists(gt.group_exposures(adpatt), \
        (skygroup1, objgroup1, skygroup2, objgroup2, skygroup3))

