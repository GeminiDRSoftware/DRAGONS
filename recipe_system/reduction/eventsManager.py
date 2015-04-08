#
#                                                                     QAP Gemini
#
#                                                              eventsManager.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
import json
import time
import re

from astrodata import AstroData

# ------------------------------------------------------------------------------
class EventsManager:
    # reject reloading events older, in secs (cur 7 days)
    lose_duration = float(24*60*60)*7

    def __init__(self, rc=None, persist=False):
        self.rc = rc
        self.event_list = []
        self.event_index = {}
        self.persist = persist   #  False or filename in .adcc
        self.persist_load()


    def _get_stacklist(self, ad):
        # Find a list of all images that went into a stack for a stacked image
        raw_rgx = re.compile(r'^(tmp)(\d+)(gemcombine)(N|S)(\d{8})(S)(\d{4})(\.fits)(\[SCI,)(\d+)(\])$') 
        # For quick reductions for the GUI, we are assuming that the same
        # images are combined for all extensions
        raw_list = ad['SCI',1].get_key_value('IMCMB***')
        if raw_list:
            stack_list = []
            for item in raw_list:
                tempfile = raw_list[item]
                if raw_rgx.match(raw_list[item]):
                    stack_list.append(''.join(re.split(raw_rgx, raw_list[item])[4:9]))
        return stack_list            


    def get_metadict(self, ad):
        # Key: metadata dictionary key, Value: descriptor name
        descriptor_dict = {"datalabel" : "data_label",
                           "local_time": "local_time",
                           "ut_time"   : "ut_datetime",
                           "wavelength": "central_wavelength",
                           "filter"    : "filter_name",
                           "waveband"  : "wavelength_band",
                           "airmass"   : "airmass",
                           "instrument": "instrument",
                           "object"    : "object",
                           "wfs"       : "wavefront_sensor",}

        options = {"central_wavelength":"asMicrometers=True",
                   "filter_name":"pretty=True",}

        postprocess = {"local_time" : '.strftime("%H:%M:%S.%f")',
                       "ut_datetime": '.strftime("%Y-%m-%d %H:%M:%S.%f")',}

        # Make the metadata dictionary.  Start with the items that
        # do not come from descriptors
        mtd_dict = {"raw_filename": ad.filename, "types": ad.types,}

        for mtd_name, desc_name in descriptor_dict.iteritems():
            if options.has_key(desc_name):
                opt = options[desc_name]
            else:
                opt = ''

            if postprocess.has_key(desc_name):
                pp = postprocess[desc_name]
            else:
                pp = ''

            try:
                exec('dv = ad.%s(%s).as_pytype()%s' % (desc_name, opt, pp))
            except:
                dv = None

            mtd_dict[mtd_name] = dv
            
        # If the file is a processed stack, then add the filenames of the 
        # data that went into the stack
        if ad.phu_get_key_value('STACKFRM'):
            stack_list = self._get_stacklist(ad)
            mtd_dict["stack"] = stack_list
        
        mtd = {"metadata": mtd_dict}
        return mtd

                            
    def append_event(self, ad=None, name=None, mdict=None, metadata=None, 
                     msgtype="qametric", persisted=False):

        if isinstance(ad, AstroData):
            if metadata != None:
                md = metadata
            else:
                md = self.get_metadict(ad)

            curtime = time.time()
            wholed = {"msgtype":msgtype, name:mdict, "timestamp":curtime}
            wholed.update(md)

        elif type(ad) == list:
            for msg in ad:
                if "timestamp" in msg:
                    msg.update({"reported_timestamp":msg["timestamp"]})
                msg.update({"timestamp":time.time()})
            self.event_list.extend(ad)
            if not persisted:
                self.persist_add(ad)    
            return

        elif type(ad) == dict:
            if "timestamp" not in ad:
                ad.update({"timestamp":time.time()})
            if "timestamp" in ad and "reported_timestamp" not in ad:
                ad.update({"reported_timestamp":ad["timestamp"]})
            wholed = ad

        else:
            raise "EVENT ARGUMENTS ERROR"

        self.event_list.append(wholed)
        if self.persist and not persisted:
            self.persist_add(wholed)
        
        timestamp = wholed["timestamp"]
        if timestamp not in self.event_index:
            self.event_index.update({timestamp:[]})

        ts_list = self.event_index[timestamp]
        ts_list.append(self.event_list.index(wholed))
        return


    def get_list(self, fromtime=None):
        if not fromtime:
            return self.event_list
        else:
            for i in range(len(self.event_list)):
                if self.event_list[i]["timestamp"] > fromtime:
                    return self.event_list[i:]
        return []


    def clear_list(self):
        self.event_list = []
        self.event_index = {}
        return


    def persist_add(self, ev=None):
        if type(ev) == list:
            if len(ev) == 0:
                return
            evlist = ev
        else:
            evlist = [ev]

        with open(".adcc/"+str(self.persist), "a+") as pfile:
            for ev in evlist:
                evstr = json.dumps(ev)
                pfile.write(evstr)
                pfile.write("\n")
        return
        
    def persist_load(self):
        if not self.persist:
            return

        LOSEDURATION = self.lose_duration
        ppath = ".adcc/" + str(self.persist)
        try:
            pfile = open(ppath)
        except IOError:
            print "em152: no persistence file " + self.persist
        else:
            with pfile:
                plines = pfile.readlines()
                i = 0
                now = time.time()
                try:
                    for line in plines:
                        i+=1
                        ev = json.loads(line)
                        qbg = "bg" in ev
                        qcc = "cc" in ev
                        qiq = "iq" in ev
                        ts = ev["timestamp"]
                        age = now - ts

                        print "AGE IN SECONDS:", age, now, ts
                        print "  em156: event #", i, ev["msgtype"],
                        print "  bg=%s, cc=%s, iq=%s" % (qbg, qcc, qiq)

                        if age < LOSEDURATION:
                            self.append_event(ev, persisted=True)
                        else:
                            print "  em183: DISCARDED"
                except ValueError, ve:
                    print "em156: valueerror", repr(dir(ve)), repr(ve.args)
        return
