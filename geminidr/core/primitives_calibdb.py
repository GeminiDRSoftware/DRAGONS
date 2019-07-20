#
#                                                                  gemini_python
#
#                                                          primitives_calibdb.py
# ------------------------------------------------------------------------------
import os
import re
from importlib import import_module

from gempy.gemini import gemini_tools as gt

from recipe_system.cal_service.calrequestlib import get_cal_requests
from recipe_system.cal_service.calrequestlib import process_cal_requests
from recipe_system.cal_service.transport_request import upload_calibration

from geminidr import PrimitivesBASE
from . import parameters_calibdb

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
REQUIRED_TAG_DICT = {'processed_arc':    ['PROCESSED', 'ARC'],
                     'processed_bias':   ['PROCESSED', 'BIAS'],
                     'processed_dark':   ['PROCESSED', 'DARK'],
                     'processed_flat':   ['PROCESSED', 'FLAT'],
                     'processed_fringe': ['PROCESSED', 'FRINGE'],
                     'bpm':              ['BPM']}
# ------------------------------------------------------------------------------
@parameter_override
class CalibDB(PrimitivesBASE):
    """
    Only 'storeProcessedXXX' calibration primitives have associated parameters.
    """
    tagset = None

    def __init__(self, adinputs, **kwargs):
        super(CalibDB, self).__init__(adinputs, **kwargs)
        self._param_update(parameters_calibdb)
        self._not_found = "Calibration not found for {}"

    def _get_cal(self, adinput, caltype):
        caloutputs = []
        adinputs = adinput if isinstance(adinput, list) else [adinput]
        for ad in adinputs:
            key = (ad, caltype)
            calib = self.calibrations[key]
            if not calib:
                caloutputs.append(None)
            # If the file isn't on disk, delete it from the dict
            # Now have to cope with calfile being a list of files
            else:
                if isinstance(calib, list):
                    cal_found = all(os.path.isfile(calfile) for calfile in calib)
                else:
                    cal_found = os.path.isfile(calib)
                if cal_found:
                    caloutputs.append(calib)
                else:
                    del self.calibrations[key]
                    self.calibrations.cache_to_disk()
                    caloutputs.append(None)
        return caloutputs if isinstance(adinput, list) else caloutputs[0]

    def _assert_calibrations(self, adinputs, caltype):
        for ad in adinputs:
            calurl = self._get_cal(ad, caltype)                 # from cache
            if not calurl and "qa" not in self.mode:
                    raise IOError(self._not_found.format(ad.filename))
        return adinputs

    def addCalibration(self, adinputs=None, **params):
        caltype = params["caltype"]
        calfile = params["calfile"]
        for ad in adinputs:
            self.calibrations[ad, caltype] = calfile
        return adinputs

    def getCalibration(self, adinputs=None, caltype=None, refresh=True,
                       howmany=None):
        """
        Uses the calibration manager to population the Calibrations dict for
        all frames, updating any existing entries
        
        Parameters
        ----------
        adinputs: list of ADs
            files for which calibrations are needed
        caltype: str
            type of calibration required (e.g., "processed_bias")
        refresh: bool
            if False, only seek calibrations for ADs without them; otherwise
            request calibrations for all ADs
        howmany: None/int
            maximum number of calibrations to return per AD (None means return
            the filename of one, rather than a list of filenames)
        """
        log = self.log
        ad_rq = adinputs if refresh else [ad for ad in adinputs
                                          if not self._get_cal(ad, caltype)]
        cal_requests = get_cal_requests(ad_rq, caltype)
        calibration_records = process_cal_requests(cal_requests, howmany=howmany)
        for ad, calfile in calibration_records.items():
            self.calibrations[ad, caltype] = calfile
        return adinputs

    def getProcessedArc(self, adinputs=None, **params):
        caltype = "processed_arc"
        self.getCalibration(adinputs, caltype=caltype, refresh=params["refresh"])
        self._assert_calibrations(adinputs, caltype)
        return adinputs

    def getProcessedBias(self, adinputs=None, **params):
        caltype = "processed_bias"
        self.getCalibration(adinputs, caltype=caltype, refresh=params["refresh"])
        self._assert_calibrations(adinputs, caltype)
        return adinputs

    def getProcessedDark(self, adinputs=None, **params):
        caltype = "processed_dark"
        self.getCalibration(adinputs, caltype=caltype, refresh=params["refresh"])
        self._assert_calibrations(adinputs, caltype)  
        return adinputs
    
    def getProcessedFlat(self, adinputs=None, **params):
        caltype = "processed_flat"
        self.getCalibration(adinputs, caltype=caltype, refresh=params["refresh"])
        self._assert_calibrations(adinputs, caltype)        
        return adinputs
    
    def getProcessedFringe(self, adinputs=None, **params):
        caltype = "processed_fringe"
        log = self.log
        self.getCalibration(adinputs, caltype=caltype, refresh=params["refresh"])
        self._assert_calibrations(adinputs, caltype)
        return adinputs

    def getMDF(self, adinputs=None):
        caltype = "mask"
        log = self.log
        inst_lookups = self.inst_lookups
        try:
            masks = import_module('.maskdb', inst_lookups)
            mdf_dict = getattr(masks, 'mdf_dict')
        except (ImportError, AttributeError):
            mdf_dict = None

        rqs_actual = [ad for ad in adinputs if self._get_cal(ad, caltype) is None]
        for ad in rqs_actual:
            if 'SPECT' in ad.tags:
                mask_name = ad.focal_plane_mask()
                key = '{}_{}'.format(ad.instrument(), mask_name)
                if mdf_dict is not None:
                    try:
                        mdf = os.path.join(os.path.dirname(masks.__file__),
                                           'MDF', mdf_dict[key])
                    except KeyError:
                        log.warning("MDF not found in {}".format(inst_lookups))
                    else:
                        self.calibrations[ad, caltype] = mdf
                        continue
                log.stdinfo("Requesting MDF from fitsstore ...")
                mdf_requests = get_cal_requests([ad], caltype)
                mdf_records = process_cal_requests(mdf_requests)
                for ad, calfile in mdf_records.items():
                    self.calibrations[ad, caltype] = calfile

        return adinputs

    # =========================== STORE PRIMITIVES =================================
    def storeCalibration(self, adinputs=None, **params):
        """
        Will write calibrations in calibrations/<cal_type>/
        """ 
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        storedcals = self.cachedict["calibrations"]
        caltype = params["caltype"]
        required_tags = REQUIRED_TAG_DICT[caltype]

        # Create storage directory if it doesn't exist
        if not os.path.exists(os.path.join(storedcals, caltype)):
            os.mkdir(os.path.join(storedcals, caltype))

        for ad in adinputs:
            if not ad.tags.issuperset(required_tags):
                log.warning("File {} is not recognized as a {}. Not storing as"
                            " a calibration.".format(ad.filename, caltype))
                continue
            fname = os.path.join(storedcals, caltype, os.path.basename(ad.filename))
            ad.write(fname, overwrite=True)
            log.stdinfo("Calibration stored as {}".format(fname))
            if self.upload and 'calibs' in self.upload:
                try:
                    upload_calibration(fname)
                except:
                    log.warning("Unable to upload file to calibration system")
                else:
                    msg = "File {} uploaded to fitsstore."
                    log.stdinfo(msg.format(os.path.basename(ad.filename)))
        return adinputs

    def _markAsCalibration(self, adinputs=None, suffix=None, update_datalab=True,
                          primname=None, keyword=None):
        """
        Updates filenames, datalabels (if asked) and adds header keyword
        prior to storing AD objects as calibrations
        """
        for ad in adinputs:
            if suffix:
                ad.update_filename(suffix=suffix, strip=True)
            if update_datalab:
                _update_datalab(ad, suffix, self.keyword_comments)
            gt.mark_history(adinput=ad, primname=primname, keyword=keyword)
        return adinputs

    def storeProcessedArc(self, adinputs=None, suffix=None, force=False):
        caltype = 'processed_arc'
        self.log.debug(gt.log_message("primitive", self.myself(), "starting"))
        if force:
            adinputs = gt.convert_to_cal_header(adinput=adinputs, caltype="arc",
                                                keyword_comments=self.keyword_comments)
        adinputs = self._markAsCalibration(adinputs, suffix=suffix,
                                    primname=self.myself(), keyword="PROCARC")
        self.storeCalibration(adinputs, caltype=caltype)
        return adinputs

    def storeProcessedBias(self, adinputs=None, suffix=None, force=False):
        caltype = 'processed_bias'
        self.log.debug(gt.log_message("primitive", self.myself(), "starting"))
        if force:
            adinputs = gt.convert_to_cal_header(adinput=adinputs, caltype="bias",
                                                keyword_comments=self.keyword_comments)
        adinputs = self._markAsCalibration(adinputs, suffix=suffix,
                                    primname=self.myself(), keyword="PROCBIAS")
        self.storeCalibration(adinputs, caltype=caltype)
        return adinputs

    def storeBPM(self, adinputs=None, suffix=None):
        caltype = 'bpm'
        self.log.debug(gt.log_message("primitive", self.myself(), "starting"))
        adinputs = gt.convert_to_cal_header(adinput=adinputs, caltype="bpm",
                                            keyword_comments=self.keyword_comments)
        adinputs = self._markAsCalibration(adinputs, suffix=suffix,
                    primname=self.myself(), update_datalab=False, keyword="BPM")
        self.storeCalibration(adinputs, caltype=caltype)
        return adinputs

    def storeProcessedDark(self, adinputs=None, suffix=None, force=False):
        caltype = 'processed_dark'
        self.log.debug(gt.log_message("primitive", self.myself(), "starting"))
        if force:
            adinputs = gt.convert_to_cal_header(adinput=adinputs, caltype="dark",
                                                keyword_comments=self.keyword_comments)
        adinputs = self._markAsCalibration(adinputs, suffix=suffix,
                                    primname=self.myself(), keyword="PROCDARK")
        self.storeCalibration(adinputs, caltype=caltype)
        return adinputs

    def storeProcessedFlat(self, adinputs=None, suffix=None, force=False):
        caltype = 'processed_flat'
        self.log.debug(gt.log_message("primitive", self.myself(), "starting"))
        if force:
            adinputs = gt.convert_to_cal_header(adinput=adinputs, caltype="flat",
                                                keyword_comments=self.keyword_comments)
        adinputs = self._markAsCalibration(adinputs, suffix=suffix,
                                    primname=self.myself(), keyword="PROCFLAT")
        self.storeCalibration(adinputs, caltype=caltype)
        return adinputs
    
    def storeProcessedFringe(self, adinputs=None, suffix=None):
        caltype = 'processed_fringe'
        self.log.debug(gt.log_message("primitive", self.myself(), "starting"))
        adinputs = gt.convert_to_cal_header(adinput=adinputs, caltype="fringe",
                                          keyword_comments=self.keyword_comments)
        adinputs = self._markAsCalibration(adinputs, suffix=suffix,
                primname=self.myself(), keyword="PROCFRNG", update_datalab=False)
        self.storeCalibration(adinputs, caltype=caltype)
        return adinputs

##################

def _update_datalab(ad, suffix, keyword_comments_lut):
    # Update the DATALAB. It should end with 'suffix'.  DATALAB will 
    # likely already have '_stack' suffix that needs to be replaced.
    searchsuffix = re.compile(r'(?<=[A-Za-z0-9\-])\_([a-z]+)')
    datalab = ad.data_label()
    new_datalab = re.sub(searchsuffix, suffix, datalab)
    if new_datalab == datalab:
        new_datalab += suffix
    ad.phu.set('DATALAB', new_datalab, keyword_comments_lut['DATALAB'])
    return
