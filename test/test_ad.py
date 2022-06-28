import astrodata
# from gemini_instruments import gmos
import igrins_instruments
import importlib
importlib.reload(igrins_instruments)

def test_ad_igrins():
   # fn = "../test/SDCH_20220301_0001.fits"
   fn = "./test/SDCH_20220301_0001.fits"
   ad = astrodata.open(fn)
   assert ad.instrument() == "IGRINS"
   assert ad.observation_class() == "partnerCal"
   assert ad.observation_type() == "FLAT"
   tags = set(["IGRINS", "RAW", "UNPREPARED", "VERSION1", "GEMINI"])
   assert tags.issubset(ad.tags)

# test_ad_igrins()
