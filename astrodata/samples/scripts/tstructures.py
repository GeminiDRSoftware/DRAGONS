from astrodata import AstroData
from astrodata.structuredslice import pixel_exts,bintable_exts

# some files for tests, of === the one used
ffile = "../../../../test_data/gmosspect/gsN20011222S027.fits"
od = "../../../../test_data/gndeploy1/N20110826S0336.fits"
of = nf = "../../../../test_data/multibins.fits"


od = AstroData(of) #original od
print od.infostr()


pixad = od[pixel_exts]
print pixad.infostr()

binad = od[bintable_exts]
print binad.infostr()

print od.gain()
for ext in od:
    try:
        print "<<"*20
        print "gain", ext.gain()
        
        print ">>"*20
    except:
        print "failed on "
        print ext.infostr()
        raise