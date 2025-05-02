#
#                                                                     QAP Gemini
#
#                                                               eventsManager.py
# ------------------------------------------------------------------------------
import json
import time
import re

from astrodata import AstroData

# ------------------------------------------------------------------------------
class EventsManager:
    def __init__(self):
        self.event_list = []
        self.event_index = {}

    def _get_stacklist(self, ad):
        # Find a list of all images that went into a stack for a stacked image
        #raw_rgx = re.compile(r'^(tmp)(\d+)(gemcombine)(N|S)(\d{8})(S)(\d{4})
        # (\.fits)(\[SCI,)(\d+)(\])$')
        # For quick reductions for the GUI, we are assuming that the same
        # images are combined for all extensions
        # Depracated:
        # raw_rgx = re.compile(r'tmp\d+gemcombine[NS]\d{8}S\d{4}\.fits\[SCI,\d+\]')

        return list(ad.phu.get('IMCMB***').values())

    def get_metadict(self, ad):
        # Key: metadata dictionary key, Value: descriptor name
        descriptor_dict = {
            "datalabel" : "data_label",
            "local_time": "local_time",
            "ut_time"   : "ut_datetime",
            "wavelength": "central_wavelength",
            "filter"    : "filter_name",
            "waveband"  : "wavelength_band",
            "airmass"   : "airmass",
            "instrument": "instrument",
            "object"    : "object",
            "wfs"       : "wavefront_sensor",
        }

        options = {
            "central_wavelength": dict(asMicrometers=True),
            "filter_name"       : dict(pretty=True),
        }

        postprocess = {
            "local_time" : lambda x: x.strftime("%H:%M:%S.%f"),
            "ut_datetime": lambda x: x.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "central_wavelength": lambda x: float(x),   # np.float32 cannot be json'ed.
        }

        mtd_dict = {
            "raw_filename": ad.filename,
            "types": list(ad.tags),
        }

        for mtd_name, desc_name in list(descriptor_dict.items()):
            try:
                descriptor = getattr(ad, desc_name)
            except AttributeError:
                mtd_dict[mtd_name] = None
            else:
                # Get the postprocessing transform. If there's none, use
                # an identity function.
                postproc = postprocess.get(desc_name, lambda x: x)
                dv = descriptor(**options.get(desc_name, {}))
                mtd_dict[mtd_name] = postproc(dv)

        # If the file is a processed stack, then add the filenames of the
        # data that went into the stack
        if ad.phu.get('STACKFRM'):
            stack_list = self._get_stacklist(ad)
            mtd_dict["stack"] = stack_list

        mtd = {"metadata": mtd_dict}
        return mtd

    def append_event(self, ad=None, name=None, mdict=None, metadata=None,
                     msgtype="qametric"):

        curtime = time.time()
        wholed = {
            "msgtype"  : msgtype,
            name       : mdict,
            "timestamp": curtime
        }

        if isinstance(ad, AstroData):
            if metadata is not None:
                md = metadata
            else:
                md = self.get_metadict(ad)
            wholed.update(md)

        elif isinstance(ad, list):
            for msg in ad:
                if "timestamp" in msg:
                    msg.update({"reported_timestamp":msg["timestamp"]})
                msg.update({"timestamp":time.time()})
            self.event_list.extend(ad)
            return

        elif isinstance(ad, dict):
            if "timestamp" not in ad:
                ad.update({"timestamp":time.time()})
            if "timestamp" in ad and "reported_timestamp" not in ad:
                ad.update({"reported_timestamp":ad["timestamp"]})
            wholed = ad

        else:
            raise TypeError("Bad Arguments")

        self.event_list.append(wholed)
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
