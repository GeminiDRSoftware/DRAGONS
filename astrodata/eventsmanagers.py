import time

class EventsManager:

    event_list = None
    event_index = None
    rc = None
    def __init__(self, rc=None):
        self.rc = rc
        self.event_list = []
        self.event_index = {}
        
    def getMetadataDict(self, ad):
        mtd = {"metadata":
                { "filename": ad.filename,
                  "datalabel": ad.data_label()
                  "local_time": ad.local_time().strftime("%Y-%m-%d %H:%M:%S"),
                  "ut_time": ad.ut_time().strftime("%Y-%m-%d %H:%M:%S"),
                  "wavelength": ad.central_wavelength(asNanometers=True),
                  "filter": ad.filter(pretty=True),
                  "waveband": ad.wavelength_band(),
                  "airmass": ad.airmass(),
                  "instrument": ad.instrument(),
                  "object": ad.object(),
                  "types": ad.types,
                }
              }
        return mtd
        
    def appendEvent(self, ad, name, mdict):
        md = self.getMetadataDict(ad)
        print repr("em23:"+repr(md))        
