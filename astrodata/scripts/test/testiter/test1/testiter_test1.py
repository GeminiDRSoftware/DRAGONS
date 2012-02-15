from astrodata.adutils.testutil import ok_, sci123, AstroData 

def test1():
    '''ASTRODATA-iter TEST 1: Compare for AD and for HDUList (MEF)'''
    ad = AstroData(sci123)
    aditerImageObjectIdList = []
    hduImageObjectIdList = []
    for a in ad:
        aditerImageObjectIdList.append(id(a.hdulist[1]))
    for phu in ad.hdulist[1:]:
        hduImageObjectIdList.append(id(phu))
    ok_(aditerImageObjectIdList == hduImageObjectIdList, \
        msg='Object ids are not the same')
    ad.close()

