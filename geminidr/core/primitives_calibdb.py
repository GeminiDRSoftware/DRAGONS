#
#                                                                  gemini_python
#
#                                                          primitives_calibdb.py
# ------------------------------------------------------------------------------
import os
import re
from importlib import import_module
import datetime

from gempy.gemini import gemini_tools as gt

from recipe_system.cal_service.calrequestlib import get_cal_requests
from recipe_system.cal_service.calrequestlib import process_cal_requests
from recipe_system.cal_service.transport_request import upload_calibration

from geminidr import PrimitivesBASE
from . import parameters_calibdb

from recipe_system.utils.decorators import parameter_override

# ------------------------------------------------------------------------------
REQUIRED_TAG_DICT = {'processed_arc': ['PROCESSED', 'ARC'],
                     'processed_bias': ['PROCESSED', 'BIAS'],
                     'processed_dark': ['PROCESSED', 'DARK'],
                     'processed_flat': ['PROCESSED', 'FLAT'],
                     'processed_fringe': ['PROCESSED', 'FRINGE'],
                     'bpm': ['BPM'],
                     'sq': [],
                     'ql': [],
                     'qa': [],
                     'processed_standard': ['PROCESSED', 'STANDARD'],
                     'processed_slitillum': ['PROCESSED', 'SLITILLUM']}


# ------------------------------------------------------------------------------
@parameter_override
class CalibDB(PrimitivesBASE):
    """
    Only 'storeProcessedXXX' calibration primitives have associated parameters.
    """
    tagset = None

    def __init__(self, adinputs, **kwargs):
        super().__init__(adinputs, **kwargs)
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
        log = self.log
        for ad in adinputs:
            calurl = self._get_cal(ad, caltype)  # from cache
            if not calurl and "sq" in self.mode:
                log.warning(self._not_found.format(ad.filename))
                #raise OSError(self._not_found.format(ad.filename))
            elif calurl:
                log.stdinfo('Using '+calurl+' for '+ad.filename)
        return adinputs

    def addCalibration(self, adinputs=None, **params):
        caltype = params["caltype"]
        calfile = params["calfile"]
        for ad in adinputs:
            self.calibrations[ad, caltype] = calfile
        return adinputs

    def getCalibration(self, adinputs=None, caltype=None, procmode=None,
                       refresh=True, howmany=None):
        """
        Uses the calibration manager to population the Calibrations dict for
        all frames, updating any existing entries

        Parameters
        ----------
        adinputs: <list>
            List of ADs of files for which calibrations are needed

        caltype: <str>
            type of calibration required (e.g., "processed_bias")

        refresh: <bool>
            if False, only seek calibrations for ADs without them; otherwise
            request calibrations for all ADs. Default is True.

        howmany: <int> or <None>
            Maximum number of calibrations to return per AD (None means return
            the filename of one, rather than a list of filenames)

        """
        log = self.log
        ad_rq = adinputs if refresh else [ad for ad in adinputs
                                          if not self._get_cal(ad, caltype)]

        # TODO refactor Calibrations out if we stick with this
        # If refresh, clear out the cache so we don't use it
        if refresh:
            for ad in ad_rq:
                del self.calibrations[ad, caltype]

        cal_requests = get_cal_requests(ad_rq, caltype, procmode)
        calibration_records = process_cal_requests(cal_requests, howmany=howmany)
        for ad, calfile in calibration_records.items():
            self.calibrations[ad, caltype] = calfile
        return adinputs

    def getProcessedArc(self, adinputs=None, **params):
        # if we are working in 'sq' mode, must retrieve 'sq' calibrations.
        # for self.mode ql and qa, just get the best matched processed
        # calibration whether it is of ql or sq quality.

        procmode = 'sq' if self.mode == 'sq' else None

        caltype = "processed_arc"
        self.getCalibration(adinputs, caltype=caltype, procmode=procmode,
                            refresh=params["refresh"])
        self._assert_calibrations(adinputs, caltype)
        return adinputs

    def getProcessedBias(self, adinputs=None, **params):
        procmode = 'sq' if self.mode == 'sq' else None
        caltype = "processed_bias"
        self.getCalibration(adinputs, caltype=caltype, procmode=procmode,
                            refresh=params["refresh"])
        self._assert_calibrations(adinputs, caltype)
        return adinputs

    def getProcessedDark(self, adinputs=None, **params):
        procmode = 'sq' if self.mode == 'sq' else None
        caltype = "processed_dark"
        self.getCalibration(adinputs, caltype=caltype, procmode=procmode,
                            refresh=params["refresh"])
        self._assert_calibrations(adinputs, caltype)
        return adinputs

    def getProcessedFlat(self, adinputs=None, **params):
        procmode = 'sq' if self.mode == 'sq' else None
        caltype = "processed_flat"
        self.getCalibration(adinputs, caltype=caltype, procmode=procmode,
                            refresh=params["refresh"])
        self._assert_calibrations(adinputs, caltype)
        return adinputs

    def getProcessedFringe(self, adinputs=None, **params):
        procmode = 'sq' if self.mode == 'sq' else None
        caltype = "processed_fringe"
        log = self.log
        self.getCalibration(adinputs, caltype=caltype, procmode=procmode,
                            refresh=params["refresh"])
        self._assert_calibrations(adinputs, caltype)
        return adinputs

    def getProcessedStandard(self, adinputs=None, **params):
        procmode = 'sq' if self.mode == 'sq' else None
        caltype = "processed_standard"
        self.getCalibration(adinputs, caltype=caltype, procmode=procmode,
                            refresh=params["refresh"])
        self._assert_calibrations(adinputs, caltype)
        return adinputs

    def getProcessedSlitIllum(self, adinputs=None, **params):
        procmode = 'sq' if self.mode == 'sq' else None
        caltype = "processed_slitillum"
        self.getCalibration(adinputs, caltype=caltype, procmode=procmode,
                            refresh=params["refresh"])
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
            mask_name = ad.focal_plane_mask()
            key = '{}_{}'.format(ad.instrument(), mask_name)
            if mdf_dict is not None:
                try:
                    filename = mdf_dict[key]

                    # Escape route to allow certain focal plane masks to
                    # not require MDFs
                    if filename is None:
                        continue
                    mdf = os.path.join(os.path.dirname(masks.__file__),
                                       'MDF', filename)
                except KeyError:
                    log.warning("MDF not found in {}".format(inst_lookups))
                else:
                    self.calibrations[ad, caltype] = mdf
                    continue
            log.stdinfo("Requesting MDF from calibration server...")
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

        # If we are one of the 'science' types, then we store it as science.
        # This changes the log messages to refer to the files as science
        # and ultimately routes to the upload_file web api instead of
        # upload_processed_cal
        is_science = caltype in ['sq', 'ql', 'qa']

        # Create storage directory if it doesn't exist
        if not os.path.exists(os.path.join(storedcals, caltype)):
            os.makedirs(os.path.join(storedcals, caltype))

        for ad in adinputs:
            if self.upload and not is_science and 'RELEASE' in ad.phu:
                # if RELEASE is in the future, reset the RELEASE keyword to
                # today, otherwise QAP and FIRE, and SCALeS will not be able
                # to pick them up.  DRAGONS v3.0. We might be able to use
                # download cookies in v3.1.
                # todo:  consider removing in 3.1
                date = datetime.datetime.strptime(ad.phu['RELEASE'], '%Y-%m-%d')
                now = datetime.datetime.now()
                if date > now:
                    ad.phu['RELEASE'] = now.strftime('%Y-%m-%d')

            if not ad.tags.issuperset(required_tags):
                log.warning("File {} is not recognized as a {}. Not storing as"
                            " {}.".format(ad.filename, caltype,
                            "science" if is_science else "a calibration"))
                continue
            fname = os.path.join(storedcals, caltype, os.path.basename(ad.filename))
            ad.write(fname, overwrite=True)
            log.stdinfo("{} stored as {}".format("Science" if is_science else "Calibration", fname))
            if self.upload and ((is_science and 'science' in self.upload) or \
                                (not is_science and 'calibs' in self.upload)):
                try:
                    upload_calibration(fname, is_science=is_science)
                except:
                    log.warning("Unable to upload file to {} system"
                                .format("science" if is_science else "calibration"))
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
            # if user mode: not uploading and sq, don't add mode.
            if self.mode is 'sq' and (not self.upload or 'calibs' not in self.upload) :
                proc_suffix = f""
            else:
                proc_suffix = f"_{self.mode}"

            if suffix:
                proc_suffix += suffix
            ad.update_filename(suffix=proc_suffix, strip=True)
            if update_datalab:
                _update_datalab(ad, suffix, self.keyword_comments)
            gt.mark_history(adinput=ad, primname=primname, keyword=keyword)
            ad.phu.set('PROCMODE', self.mode)
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

        # We only need to do this if we're uploading to the archive so the OBSTYPE
        # is set to FRINGE and OBSID, etc., are obscured. The frame will be tagged
        # as FRINGE and available locally.
        if self.upload and 'calibs' in self.upload:
            adinputs = gt.convert_to_cal_header(adinput=adinputs, caltype="fringe",
                                                keyword_comments=self.keyword_comments)
        adinputs = self._markAsCalibration(adinputs, suffix=suffix,
                                           primname=self.myself(), keyword="PROCFRNG", update_datalab=False)
        self.storeCalibration(adinputs, caltype=caltype)
        return adinputs

    def storeProcessedScience(self, adinputs=None, suffix=None):
        if self.mode not in ['sq', 'ql', 'qa']:
            self.log.warning('Mode %s not recognized in storeScience, not saving anything' % self.mode)
            return adinputs

        for ad in adinputs:
            gt.mark_history(adinput=ad, primname=self.myself(), keyword="PROCSCI")
            if suffix:
                ad.update_filename(suffix=suffix, strip=True)
            else:  # None.  Keep the one it has now.  (eg. 'stack' for imaging)
                if '_' in ad.filename:
                    suffix = '_'+ad.filename.split('_')[-1].split('.')[0]
                else:
                    suffix = ''

            ad.phu.set('PROCMODE', self.mode)

            if self.mode != 'qa' and self.upload and 'science' in self.upload:
                old_filename = ad.filename
                ad.update_filename(suffix=f"_{self.mode}"+suffix, strip=True)
                ad.write(overwrite=True)
                try:
                    upload_calibration(ad.filename, is_science=True)
                except:
                    self.log.warning("Unable to upload file to science system")
                else:
                    msg = "File {} uploaded to fitsstore."
                    self.log.stdinfo(msg.format(os.path.basename(ad.filename)))
                # Rename file on disk to avoid writing twice
                os.replace(ad.filename, old_filename)
                ad.filename = old_filename
            else:
                ad.write(overwrite=True)

        return adinputs

    def storeProcessedStandard(self, adinputs=None, suffix=None):
        caltype = 'processed_standard'
        self.log.debug(gt.log_message("primitive", self.myself(), "starting"))
        adoutputs = list()
        for ad in adinputs:
            passes = all(hasattr(ext, 'SENSFUNC') for ext in ad)
            # if all of the extensions on this ad have a sensfunc attribute:
            if passes:
                procstdads = self._markAsCalibration([ad], suffix=suffix,
                                           primname=self.myself(), keyword="PROCSTND")
                adoutputs.extend(procstdads)
            else:
                adoutputs.append(ad)
        self.storeCalibration(adinputs, caltype=caltype)
        return adoutputs

    def storeProcessedSlitIllum(self, adinputs=None, suffix=None):
        """
        Stores the Processed Slit Illumination file.

        Parameters
        ----------
        adinputs : list of AstroData
            Data that contain the Slit Illumination Response Function.
        suffix : str
            Suffix to be added to each of the input files.

        Returns
        -------
        list of AstroData : the input data is simply forwarded.
        """
        caltype = 'processed_slitillum'
        self.log.debug(gt.log_message("primitive", self.myself(), "starting"))
        adoutputs = list()
        for ad in adinputs:
            passes = 'MAKESILL' in ad.phu
            if passes:
                procstdads = self._markAsCalibration([ad], suffix=suffix,
                                                     primname=self.myself(), keyword="PROCILLM")
                adoutputs.extend(procstdads)
            else:
                adoutputs.append(ad)
        self.storeCalibration(adinputs, caltype=caltype)
        return adoutputs


##################

def _update_datalab(ad, suffix, keyword_comments_lut):
    # Update the DATALAB. It should end with 'suffix'.  DATALAB will
    # likely already have '_stack' suffix that needs to be replaced.

    # replace the _ with a - to match fitsstore datalabel standard
    # or add the - if "suffix" doesn't have a leading _
    if suffix[0] == '_':
        extension = suffix.replace('_', '-', 1).upper()
    else:
        extension = '-'+suffix.upper()

    datalab = ad.data_label()
    new_datalab = re.sub(r'-[a-zA-Z]+$', '', datalab) + extension
    ad.phu.set('DATALAB', new_datalab, keyword_comments_lut['DATALAB'])
    return
