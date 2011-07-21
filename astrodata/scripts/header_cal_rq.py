import sys
sys.path.append("/opt/gemini_python")
from astrodata import AstroData
import urllib2, urllib

# This is a GMOS_N imaging science dataset
ad = AstroData("calsearch/N20110531S0114.fits")

desc_dict = {'instrument':ad.instrument().for_db(),
             'data_label':ad.data_label().for_db(),
             'detector_x_bin':ad.detector_x_bin().for_db(),
             'detector_y_bin':ad.detector_y_bin().for_db(),
             'read_speed_setting':ad.read_speed_setting().for_db(),
             'gain_setting':ad.gain_setting().for_db(),
             'amp_read_area':ad.amp_read_area(asDict=True).for_db(),
             'ut_datetime':str(ad.ut_datetime().for_db()),
             'exposure_time':ad.exposure_time().for_db(),
             'nodandshuffle':ad.is_type('GMOS_NODANDSHUFFLE'),
             'observation_type': ad.observation_type().for_db(),
             'spectroscopy': ad.is_type("SPECT"),
             'object': ad.object().for_db()
             
             }
print repr(desc_dict)
type_list = ad.types
ad.close()

sequence = [('descriptors', desc_dict), ('types', type_list)]
postdata = urllib.urlencode(sequence)

#postdata = urllib.urlencode({"hello":1.})
url = "http://hbffits2.hi.gemini.edu/calmgr/bias/"
#url = "http://hbffits1/calmgr/processed_bias/"
# u = urllib.urlopen(url, postdata)
try:
    rq = urllib2.Request(url)
    u = urllib2.urlopen(rq, postdata)
except urllib2.HTTPError, error:
    contens = error.read()
    print contens
    sys.exit()
    
print u.read()
u.close()

