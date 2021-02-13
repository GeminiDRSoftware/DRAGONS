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

    def _assert_calibrations(self, adinputs, cals):
        log = self.log
        for ad, (calfile, origin) in zip(adinputs, cals.items()):
            log.stdinfo(f"{ad.filename} has received calibration {calfile} "
                        f"from {origin}")
        return adinputs

    def setCalibration(self, adinputs=None, **params):
        """
        Manually assigns a calibration to one or more frames. This is expected
        to only affect the UserDB, since other databases do not have a way to
        override the calibration association rules in isolation.

        Parameters
        ----------
        adinputs : <list>
            List of ADs of files for which calibrations are needed

        caltype : <str>
            type of calibration required (e.g., "processed_bias")

        calfile : <str>
            filename of calibration
        """
        self.caldb.set_calibrations(adinputs, **params)
        return adinputs

    def getProcessedArc(self, adinputs=None):
        # if we are working in 'sq' mode, must retrieve 'sq' calibrations.
        # for self.mode ql and qa, just get the best matched processed
        # calibration whether it is of ql or sq quality.
        procmode = 'sq' if self.mode == 'sq' else None
        cals = self.caldb.get_processed_arc(adinputs, procmode=procmode)
        self._assert_calibrations(adinputs, cals)
        return adinputs

    def getProcessedBias(self, adinputs=None):
        procmode = 'sq' if self.mode == 'sq' else None
        cals = self.caldb.get_processed_bias(adinputs, procmode=procmode)
        self._assert_calibrations(adinputs, cals)
        return adinputs

    def getProcessedDark(self, adinputs=None):
        procmode = 'sq' if self.mode == 'sq' else None
        cals = self.caldb.get_processed_dark(adinputs, procmode=procmode)
        self._assert_calibrations(adinputs, cals)
        return adinputs

    def getProcessedFlat(self, adinputs=None):
        procmode = 'sq' if self.mode == 'sq' else None
        cals = self.caldb.get_processed_flat(adinputs, procmode=procmode)
        self._assert_calibrations(adinputs, cals)
        return adinputs

    def getProcessedFringe(self, adinputs=None):
        procmode = 'sq' if self.mode == 'sq' else None
        cals = self.caldb.get_processed_fringe(adinputs, procmode=procmode)
        self._assert_calibrations(adinputs, cals)
        return adinputs

    def getProcessedStandard(self, adinputs=None):
        procmode = 'sq' if self.mode == 'sq' else None
        cals = self.caldb.get_processed_standard(adinputs, procmode=procmode)
        self._assert_calibrations(adinputs, cals)
        return adinputs

    def getProcessedSlitIllum(self, adinputs=None):
        procmode = 'sq' if self.mode == 'sq' else None
        cals = self.caldb.get_processed_slitillum(adinputs, procmode=procmode)
        self._assert_calibrations(adinputs, cals)
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
    def storeCalibration(self, adinputs=None, caltype=None):
        """
        Farm some calibration ADs out to the calibration database(s) to process.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        for ad in adinputs:
            self.caldb.store_calibration(ad, caltype=caltype)
        return adinputs

    def _markAsCalibration(self, adinputs=None, suffix=None, update_datalab=True,
                           primname=None, keyword=None):
        """
        Updates filenames, datalabels (if asked) and adds header keyword
        prior to storing AD objects as calibrations
        """
        for ad in adinputs:
            # if user mode: not uploading and sq, don't add mode.
            if self.mode == 'sq' and (not self.upload or 'calibs' not in self.upload) :
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
            ad.update_filename(suffix=suffix, strip=True)
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
