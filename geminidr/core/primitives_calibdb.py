#
#                                                                  gemini_python
#
#                                                          primitives_calibdb.py
# ------------------------------------------------------------------------------
import os
import re

from gempy.gemini import gemini_tools as gt

from geminidr import PrimitivesBASE
from . import parameters_calibdb

from recipe_system.utils.decorators import parameter_override, capture_provenance


# ------------------------------------------------------------------------------
@parameter_override
@capture_provenance
class CalibDB(PrimitivesBASE):
    """
    Only 'storeProcessedXXX' calibration primitives have associated parameters.
    """
    tagset = None

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_calibdb)
        self._not_found = "Calibration not found for {}"

    def _assert_calibrations(self, adinputs, cals):
        log = self.log
        for ad, (calfile, origin) in zip(adinputs, cals.items()):
            if calfile:
                log.stdinfo(f"{ad.filename}: received calibration {calfile} "
                            f"from {origin}")
            else:
                log.warning(f"{ad.filename}: NO CALIBRATION RECEIVED")
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
        cals = self.caldb.get_calibrations(adinputs, caltype="mask")
        self._assert_calibrations(adinputs, cals)
        return adinputs

    # =========================== STORE PRIMITIVES =================================
    def storeCalibration(self, adinputs=None, caltype=None):
        """
        Farm some calibration ADs out to the calibration database(s) to process.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        for ad in adinputs:
            # todo: If RELEASE for all calibration is still not set to "today"
            #       when we get to use this version for Ops, add a reset of
            #       that keyword like we did in release/3.0.x.  Otherwise
            #       SCALeS and FIRE won't be able to see new calibrations.

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
        for ad in adinputs:
            gt.mark_history(adinput=ad, primname=self.myself(), keyword="PROCSCI")
            if suffix:
                ad.update_filename(suffix=suffix, strip=True)
            else:  # None.  Keep the one it has now.  (eg. 'stack' for imaging)
                # Got to do a bit of gymnastic to figure what the current
                # suffix is.  If orig.filename and filename are equal and have
                # `_`, I have to assume that the last `_` is a suffix.  (KL)
                root, filetype = os.path.splitext(ad.orig_filename)
                if ad.orig_filename == ad.filename:
                    pre, post = ad.orig_filename.rsplit('_', 1)
                    suffix, filetype = os.path.splitext(post)
                    suffix = '_' + suffix
                else:
                    m = re.match('(.*){}(.*)'.format(re.escape(root)), ad.filename)
                    if m.groups()[1] and m.groups()[1] != filetype:

                        suffix, filetype = os.path.splitext(m.groups()[1])
                    else:
                        suffix = ''

            ad.phu.set('PROCMODE', self.mode)
            ad.write(overwrite=True)

        if self.mode not in ['sq', 'ql', 'qa']:
            self.log.warning(f'Mode {self.mode} not recognized in '
                             'storeScience, not storing anything')
            return adinputs

        # This logic will be handled by the CalDB objects, but check here to
        # avoid changing and resetting filenames
        if self.mode != 'qa' and self.upload and 'science' in self.upload:
            for ad in adinputs:
                old_filename = ad.filename
                ad.update_filename(suffix=f"_{self.mode}"+suffix, strip=True)
                self.caldb.store_calibration(ad, caltype="processed_science")
                ad.filename = old_filename

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
