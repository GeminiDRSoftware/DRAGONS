#
#                                                                  gemini_python
#
#                                                          primitives_calibdb.py
# ------------------------------------------------------------------------------
import os
import re

import astrodata
import gemini_instruments

from gempy.gemini import gemini_tools as gt

from recipe_system.cal_service.calrequestlib import get_cal_requests
from recipe_system.cal_service.calrequestlib import process_cal_requests
from recipe_system.cal_service.transport_request import upload_calibration

from recipe_system.utils.decorators import parameter_override

from parameters_calibdb import ParametersCalibration

from geminidr import PrimitivesBASE
# ------------------------------------------------------------------------------
@parameter_override
class Calibration(PrimitivesBASE):
    """
    Only 'storeProcessedXXX' calibration primitives have associated parameters.

    """
    tagset = None

    def __init__(self, adinputs, context, upmeterics=False, ucals=None, uparms=None):
        super(Calibration, self).__init__(adinputs, context, ucals=ucals, uparms=uparms)
        self.parameters = ParametersCalibration
        self._not_found = "Calibration not found for {}"

    def _add_cal(self, crecords):
        self.calibrations.update(crecords)
        caches.save_cache(self.calibrations, caches.calindfile)
        return

    def _get_cal(self, ad, caltype):
        key = (ad.data_label(), caltype)
        try:
            return self.calibrations[key][1]    # check file is
        except KeyError:
            return None

    def _assert_calibrations(self, adinputs, caltype):
        for ad in adinputs:
            calurl = self._get_cal(ad, caltype)                 # from cache
            if not calurl and "qa" not in self.context:
                    raise IOError(self._not_found.format(ad.filename))
        return adinputs

    def getCalibration(self, adinputs=None, stream='main', **params):
        caltype = params.get('caltype')
        log = self.log
        if caltype is None:
            log.error("getCalibration: Received no caltype")
            raise TypeError("getCalibration: Received no caltype.")

        rqs_actual = [ad for ad in adinputs if self._get_cal(ad, caltype) is None]
        cal_requests = get_cal_requests(rqs_actual, caltype)
        calibration_records = process_cal_requests(cal_requests)
        self._add_cal(calibration_records)
        return adinputs

    def getProcessedArc(self, adinputs=None, stream='main', **params):
        caltype = "processed_arc"
        self.getCalibration(adinputs, caltype=caltype)
        self._assert_calibrations(adinputs, caltype)
        return adinputs

    def getProcessedBias(self, adinputs=None, stream='main', **params):
        caltype = "processed_bias"
        self.getCalibration(adinputs, caltype=caltype)
        self._assert_calibrations(adinputs, caltype)
        return adinputs

    def getProcessedDark(self, adinputs=None, stream='main', **params):
        caltype = "processed_dark"
        self.getCalibration(adinputs, caltype=caltype)
        self._assert_calibrations(adinputs, caltype)  
        return adinputs
    
    def getProcessedFlat(self, adinputs=None, stream='main', **params):
        caltype = "processed_flat"
        self.getCalibration(adinputs, caltype=caltype)
        self._assert_calibrations(adinputs, caltype)        
        return adinputs
    
    def getProcessedFringe(self, adinputs=None, stream='main', **params):
        caltype = "processed_fringe"
        log = self.log
        self.getCalibration(adinputs, caltype=caltype)
        # Fringe correction is always optional; do not raise errors if fringe
        # not found
        try:
            self._assert_calibrations(adinputs, caltype)
        except IOError:
            wstr = "Warning: one or more processed fringe frames could not"
            wstr += " be found. "
            log.warn(wstr)
        return adinputs

# =========================== STORE PRIMITIVES =================================
    def storeCalibration(self, adinputs=None, stream='main', **params):
        """
        Will write calibrations in calibrations/<cal_type>/

        """ 
        caltype = params.get('caltype')
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        storedcals = self.cachedict["calibrations"]
        for ad in adinputs:
            fname = os.path.join(storedcals, caltype, os.path.basename(ad.filename))
            ad.write(filename=fname, clobber=True)
            log.stdinfo("Calibration stored as {}".format(fname))
            if self.upload_calibrations:       # !!! This does not exist!!!
                                               # OR  if 'uplaod' in self.context ??
                try:
                    upload_calibration(fname)
                except:
                    log.warning("Unable to upload file to calibration system")
                else:
                    msg = "File {} uploaded to fitsstore."
                    log.stdinfo(msg.format(os.path.basename(ad.filename)))

        return adinputs

    def storeProcessedArc(self, adinputs=None, stream='main', **params):
        caltype = 'processed_arc'
        log = self.log
        parset = getattr(self.parameters, self.myself())
        sfx = getattr(parset, 'suffix')
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        for ad in adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            _update_datalab(ad, sfx, self.keyword_comments)            
            gt.mark_history(adinput=ad, primname=self.myself(), keyword="PROCARC")

        self.storeCalibration(adinputs, caltype=caltype)
        return adinputs

    def storeProcessedBias(self, adinputs=None, stream='main', **params):
        caltype = 'processed_bias'
        log = self.log
        parset = getattr(self.parameters, self.myself())
        sfx = getattr(parset, 'suffix')
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        for ad in adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            _update_datalab(ad, sfx, self.keyword_comments)            
            gt.mark_history(adinput=ad, primname=self.myself(), keyword="PROCBIAS")
        
        self.storeCalibration(adinputs, caltype=caltype)
        return adinputs

    def storeBPM(self, adinputs=None, stream='main', **params):
        caltype = 'bpm'
        log = self.log
        sfx = "_bpm"
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        for ad in adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            gt.mark_history(adinput=ad, primname=self.myself(), keyword="BPM")

        self.storeCalibration(adinputs, caltype)
        return adinputs

    def storeProcessedDark(self, adinputs=None, stream='main', **params):
        caltype = 'processed_dark'
        log = self.log
        parset = getattr(self.parameters, self.myself())
        sfx = getattr(parset, 'suffix')
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        for ad in adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            _update_datalab(ad, sfx, self.keyword_comments)            
            gt.mark_history(adinput=ad, primname=self.myself(), keyword="PROCDARK")
        
        self.storeCalibration(adinputs, caltype=caltype)
        return adinputs
    
    def storeProcessedFlat(self, adinputs=None, stream='main', **params):
        caltype = 'processed_flat'
        log = self.log
        parset = getattr(self.parameters, self.myself())
        sfx = getattr(parset, 'suffix')
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        for ad in adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            _update_datalab(ad, sfx, self.keyword_comments)            
            gt.mark_history(adinput=ad, primname=self.myself(), keyword="PROCFLAT")

        self.storeCalibration(adinputs, caltype=caltype)
        return adinputs       
    
    def storeProcessedFringe(self, adinputs=None, stream='main', **params):
        caltype = 'processed_fringe'
        log = self.log
        parset = getattr(self.parameters, self.myself())
        sfx = getattr(parset, 'suffix')
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        for ad in adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            ad = gt.convert_to_cal_header(adinput=ad, caltype="fringe", 
                                          keyword_comments=self.keyword_comments)[0]

            gt.mark_history(adinput=ad, primname=self.myself(), keyword="PROCFRNG")
        
        self.storeCalibration(adinputs, caltype)
        return adinputs

##################

def _update_datalab(ad, suffix, keyword_comments_lut):
    # Update the DATALAB. It should end with 'suffix'.  DATALAB will 
    # likely already have '_stack' suffix that needs to be replaced.
    searchsuffix = re.compile(r'(?<=[A-Za-z0-9\-])\_([a-z]+)')
    datalab = ad.phu_get_key_value("DATALAB")
    new_datalab = re.sub(searchsuffix, suffix, datalab)
    if new_datalab == datalab:
        new_datalab += suffix
    gt.update_key(adinput=ad, keyword="DATALAB", value=new_datalab,
                  comment=None, extname="PHU", 
                  keyword_comments=keyword_comments_lut)

    return
