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
                  "wavelength": wlen[wlen_ind],
                  "waveband": wlen[wlen_ind+1],
                  "airmass": 1.063,
                  "instrument": "GMOS-N",
                  "object": "M13",
                  "types": ["GEMINI_NORTH", "GMOS_N", "GMOS_IMAGE",
                            "GEMINI", imtype, "GMOS", "GMOS_RAW",
                            "UNPREPARED", "RAW"], 
                }
              }
        return mtd
        
    def appendEvent(self, ad, name, mdict):
        md = self.getMetadataDict(ad)
        print repr("em23:"+repr(md))        
