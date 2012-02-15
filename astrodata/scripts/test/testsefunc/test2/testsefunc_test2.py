from astrodata.adutils.testutil import AstroData, eq_, sci1
from numpy import array

def test2():
    '''ASTRODATA-single-extfuncs TEST 2: set_data()  
    '''
    print('\n\t* single ext fits testfile: %s' % sci1)
    ad = AstroData(sci1)
    print('\tad = AstroData(sci1)')
    a = array([1,2,3])
    print('\ta = array([1,2,3])')
    ad.set_data(a)
    print('\tad.set_data(a)')
    for i in range(len(ad)):
        eq_(ad.data[i], a[i], msg='array elements are different')
