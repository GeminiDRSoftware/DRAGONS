import astropy.io.fits as pyfits
import astrodata
# from gemini_instruments import gmos
import igrins_instruments
import importlib
importlib.reload(igrins_instruments)

fn = "./test/SDCH_20190412_0021_thresholdFlatfielded.fits"
fn = "./test/SDCH_20190412_0011_bpm.fits"
fn = "./test/20190412/SDCH_20190412_0040.fits"
# fn = "./test/calibrations/processed_flat/SDCH_20190412_0021_flat.fits"
ad = astrodata.open(fn)
hdul = pyfits.open(fn)

print(ad.tags)
print("observation_class", ad.observation_class())

def test_ad_igrins():
   # fn = "../test/SDCH_20220301_0001.fits"
   fn = "./test/20190412/SDCH_20190412_0001.fits"
   ad = astrodata.open(fn)
   assert ad.instrument() == "IGRINS"
   assert ad.observation_class() == "partnerCal"
   assert ad.observation_type() == "FLAT_OFF"
   tags = set(["IGRINS", "RAW", "UNPREPARED", "VERSION1", "GEMINI"])
   assert tags.issubset(ad.tags)

# test_ad_igrins()
