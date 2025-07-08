import astropy.io.fits as pyfits
import astrodata
# from gemini_instruments import gmos
import igrins_instruments
import importlib
importlib.reload(igrins_instruments)
from contextlib import suppress

fn = "./test/SDCH_20190412_0021_thresholdFlatfielded.fits"
fn = "./test/SDCH_20190412_0011_bpm.fits"
fn = "./test/20190412/SDCH_20190412_0040.fits"
# fn = "./test/calibrations/processed_flat/SDCH_20190412_0021_flat.fits"
fn = "sample_flatoff/N20240429S0365_H.fits"
fn = "unbundled_20240429/N20240429S0190_H.fits"
ad = astrodata.open(fn)
hdul = pyfits.open(fn)

print(ad[0].exposure_time())
print(ad.tags)
print("observation_class", ad.observation_class())

def test_keyword(self, name):
   for cls in self.__class__.mro():
      print(f'_{cls.__name__}__keyword_dict')
      if hasattr(self, f'_{cls.__name__}__keyword_dict'):
         print("oo", getattr(cls, f'_{cls.__name__}__keyword_dict'))
      with suppress(AttributeError, KeyError):
          # __keyword_dict is a mangled variable
          return getattr(self, f'_{cls.__name__}__keyword_dict')[name]
   else:
      raise AttributeError(f"No match for '{name}'")

test_keyword(ad, "exposure_time")

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
