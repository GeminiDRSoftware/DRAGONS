from astrodata.adutils.testutil import ok_, sci1, AstroData 

def test2():
    '''ASTRODATA-iter TEST 2: Compare for AD and for HDUList (SEF)'''
    ad = AstroData(sci1)
    aditerImageObjectIdList = []
    hduImageObjectIdList = []
    for a in ad:
        aditerImageObjectIdList.append(id(a.hdulist[1]))
    for phu in ad.hdulist[1:]:
        hduImageObjectIdList.append(id(phu))
    ok_(aditerImageObjectIdList == hduImageObjectIdList, \
        msg='Object ids are not the same')
    ad.close()

