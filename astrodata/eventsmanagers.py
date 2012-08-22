import time
from astrodata import AstroData
import datetime

class EventsManager:

    event_list = None
    event_index = None
    rc = None
    def __init__(self, rc=None):
        self.rc = rc
        self.event_list = []
        self.event_index = {}
        
    def get_metadict(self, ad):
        mtd = {"metadata":
                { "raw_filename": ad.filename,
                  "datalabel": ad.data_label().as_pytype(),
                  "local_time": ad.local_time().as_pytype().strftime("%H:%M:%S.%f"),
                  "ut_time": ad.ut_datetime().as_pytype().strftime("%Y-%m-%d %H:%M:%S.%f"),
                  "wavelength": ad.central_wavelength(asMicrometers=True).as_pytype(),
                  "filter": ad.filter_name(pretty=True).as_pytype(),
                  "waveband": ad.wavelength_band().as_pytype(),
                  "airmass": ad.airmass().as_pytype(),
                  "instrument": ad.instrument().as_pytype(),
                  "object": ad.object().as_pytype(),
                  "wfs": ad.wavefront_sensor().as_pytype(),
                  "types": ad.get_types(),
                }
              }
        return mtd
        
    def append_event(self, ad = None, name=None, mdict=None, 
                     metadata = None, msgtype="qametric"):
        # print "em32:"+repr(metadata)
        if isinstance(ad, AstroData):
            if metadata != None:
                md = metadata
            else:
                md = self.get_metadict(ad)
            curtime = time.time()
            wholed = {  
                        "msgtype":msgtype,
                        name : mdict,
                        "timestamp": curtime
                     }
            wholed.update(md)
        elif type(ad) == list:
            for msg in ad:
                if "timestamp" in msg:
                    msg.update({"reported_timestamp":msg["timestamp"]})
                msg.update({"timestamp":time.time()})
            self.event_list.extend( ad)
            return
        elif type(ad) == dict:
            if timestamp in ad:
                ad.update({"reported_timestamp":ad["timestamp"]})
            ad.update({"timestamp":time.time()})
            wholed = ad
        else:
            raise "EVENT ARGUMENTS ERROR"
        import pprint
        # print "em38:"+pprint.pformat(wholed)
        self.event_list.append(wholed)
        timestamp = wholed["timestamp"]
        # print "em38:timestamp %f" % timestamp
        if timestamp not in self.event_index:
            self.event_index.update({timestamp:[]})
        ts_list = self.event_index[timestamp]
        ts_list.append(self.event_list.index(wholed))
        
        
    def get_list(self, fromtime = None):
        if fromtime == None:
            #print "em61: send whole events list"
            return self.event_list
        # elif fromtime in self.event_index:
        #    starti = self.event_index[fromtime] + 1
        #    print "em65: index search from item #%d" % starti
        #    return self.event_list[starti:]
        else:
            # print "em83: fromtime=%f" % fromtime
            starti = 0
            for i in range(0, len(self.event_list)):
                if self.event_list[i]["timestamp"] > fromtime:
                    # print "em71: slow search from item #%d" % i
                    return self.event_list[i:]
            return []
    def clear_list(self):
        self.event_list = []
        self.event_index = {}
