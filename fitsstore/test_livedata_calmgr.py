import sys
sys.path.append("/opt/gemini_python")
from astrodata import AstroData
import urllib

# This is a GMOS_N imaging science dataset
ad = AstroData("/net/wikiwiki/dataflow/N20100820S0241.fits")

desc_dict = {'instrument':ad.instrument(),
             'data_label':ad.data_label(),
             'detector_x_bin':ad.detector_x_bin(),
             'detector_y_bin':ad.detector_y_bin(),
             'read_speed_setting':ad.read_speed_setting(),
             'gain_setting':ad.gain_setting(),
             'amp_read_area':ad.amp_read_area(asDict=True),
             'ut_datetime':ad.ut_datetime(),
             'exposure_time':ad.exposure_time(),
             'nodandshuffle':ad.isType('GMOS_NODANDSHUFFLE')
             }
type_list = ad.types
ad.close()

sequence = (('descriptors', desc_dict), ('types', type_list))
postdata = urllib.urlencode(sequence)

url = "http://hbffits3/calmgr/processed_bias/"
#url = "http://hbffits1/calmgr/processed_bias/"
u = urllib.urlopen(url, postdata)
print u.read()
u.close()

