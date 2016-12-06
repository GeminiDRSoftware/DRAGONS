#
#                                                                  gemini_python
#
#                                                          primitives_calibdb.py
# ------------------------------------------------------------------------------
import os
import re

import astrodata
import gemini_instruments

from gempy.utils  import logutils
from gempy.gemini import gemini_tools as gt

from recipe_system.stacks import IDFactory

from recipe_system.cal_service.calrequestlib import get_cal_requests
from recipe_system.cal_service.calrequestlib import process_cal_requests
from recipe_system.cal_service.transport_request import upload_calibration

from geminidr import PrimitivesBASE

# ------------------------------------------------------------------------------
class Calibration(PrimitivesBASE):
    """
    There are no parameters associated with any calibration primitives.

    """
    tagset = None

    def __init__(self, adinputs, context, upmeterics=False, ucals=None, uparms=None):
        super(Calibration, self).__init__(adinputs, context, ucals=ucals, uparms=uparms)
        self.parameters = None
        self._not_found = "Calibration not found for {}"

    def _assert_calibrations(self, adinputs, caltype):
        for ad in adinputs:
            calurl = self.get_cal(ad, caltype)                   #get from cache
            if not calurl and "qa" not in self.context:
                    raise IOError(self._not_found.format(ad.filename))
        return adinputs

    def getCalibration(self, adinputs=None, stream='main', **params):
        caltype = params.get('caltype')
        log = self.log
        if caltype is None:
            log.error("getCalibration: Received no caltype")
            raise TypeError("getCalibration: Received no caltype.")

        cal_requests = get_cal_request(adinputs, caltype)
        calibration_records = process_cal_requests(cal_requests)
        self._add_cal(calibration_records)
        return adinputs

    def getProcessedArc(self, adinputs=None, stream='main', **params):
        caltype = "processed_arc"
        log = self.log
        self.getCalibration(adinputs, caltype=caltype)
        self._assert_calibrations(adinputs, caltype)
        return adinputs

    def getProcessedBias(self, adinputs=None, stream='main', **params):
        caltype = "processed_bias"
        log = self.log
        self.getCalibration(adinputs, caltype=caltype)
        self._assert_calibrations(adinputs, caltype)
        return adinputs

    def getProcessedDark(self, adinputs=None, stream='main', **params):
        caltype = "processed_dark"
        log = self.log
        self.getCalibration(adinputs, caltype=caltype)
        self._assert_calibrations(adinputs, caltype)  
        return adinputs
    
    def getProcessedFlat(self, adinputs=None, stream='main', **params):
        caltype = "processed_flat"
        log = self.log
        self.getCalibration(adinputs, caltype=caltype)
        self._assert_calibrations(adinputs, caltype)        
        return adinputs
    
    def getProcessedFringe(self, adinputs=None, stream='main', **params):
        caltype = "processed_fringe"
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





# =========================== STORE PRIMITIVES ================================
    def storeCalibration(self, adinputs=None, stream='main', **params):
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeCalibration", "starting"))
        
        # Determine the path where the calibration will be stored
        storedcals = rc["cachedict"]["storedcals"]
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Construct the filename of the calibration, including the path
            fname = os.path.join(storedcals, os.path.basename(ad.filename))
            
            # Write the calibration to disk. Use rename=False so that
            # ad.filename does not change (i.e., does not include the
            # calibration path)
            ad.write(filename=fname, rename=False, clobber=True)
            log.stdinfo("Calibration stored as %s" % fname)
            
            if "upload" in rc.context:
                try:
                    upload_calibration(fname)
                except:
                    log.warning("Unable to upload file to calibration system")
                else:
                    log.stdinfo("File %s uploaded to fitsstore." % 
                                os.path.basename(ad.filename))
            yield rc
        
        yield rc
    
    def storeProcessedArc(self, adinputs=None, stream='main', **params):
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeProcessedArc",
                                 "starting"))
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)

            # Update DATALAB
            # The self.keyword_comments is required because "stupid".
            _update_datalab(ad, rc['suffix'], self.keyword_comments)            
            
            # Adding a PROCARC time stamp to the PHU
            gt.mark_history(adinput=ad, primname=self.myself(), keyword="PROCARC")
            
            # Refresh the AD types to reflect new processed status
            ad.refresh_types()
        
        # Upload arc(s) to cal system
        rc.run("storeCalibration")
        
        yield rc
    
    def storeProcessedBias(self, adinputs=None, stream='main', **params):
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeProcessedBias",
                                 "starting"))
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)
            
            # Update DATALAB
            # The self.keyword_comments is required because "stupid".
            _update_datalab(ad, rc['suffix'], self.keyword_comments)            
            
            # Adding a PROCBIAS time stamp to the PHU
            gt.mark_history(adinput=ad, primname=self.myself(), keyword="PROCBIAS")
            
            # Refresh the AD types to reflect new processed status
            ad.refresh_types()
        
        # Upload bias(es) to cal system
        rc.run("storeCalibration")
        
        yield rc

    def storeBPM(self, adinputs=None, stream='main', **params):
        # Instantiate the log
        log = logutils.get_logger(__name__)

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeBPM","starting"))

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():

            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.filename_updater(adinput=ad, suffix="_bpm",
                                              strip=True)

            # Adding a BPM time stamp to the PHU
            gt.mark_history(adinput=ad, primname=self.myself(), keyword="BPM")

            # Refresh the AD types to reflect new processed status
            ad.refresh_types()

        # Upload to cal system
        rc.run("storeCalibration")

        yield rc

    def storeProcessedDark(self, adinputs=None, stream='main', **params):
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeProcessedDark",
                                 "starting"))
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)

            # Update DATALAB
            # The self.keyword_comments is required because "stupid".
            _update_datalab(ad, rc['suffix'], self.keyword_comments)            
            
            # Adding a PROCDARK time stamp to the PHU
            gt.mark_history(adinput=ad, primname=self.myself(), keyword="PROCDARK")
            
            # Refresh the AD types to reflect new processed status
            ad.refresh_types()
        
        # Upload to cal system
        rc.run("storeCalibration")
        
        yield rc
    
    def storeProcessedFlat(self, adinputs=None, stream='main', **params):
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeProcessedFlat",
                                 "starting"))
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)

            # Update DATALAB
            # The self.keyword_comments is required because "stupid".
            _update_datalab(ad, rc['suffix'], self.keyword_comments)            
            
            # Adding a PROCFLAT time stamp to the PHU
            gt.mark_history(adinput=ad, primname=self.myself(), keyword="PROCFLAT")
            
            # Refresh the AD types to reflect new processed status
            ad.refresh_types()
        
        # Upload to cal system
        rc.run("storeCalibration")
        
        yield rc
    
    def storeProcessedFringe(self, adinputs=None, stream='main', **params):
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeProcessedFringe",
                                 "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)
            
            # Sanitize the headers of the file so that it looks like
            # a public calibration file rather than a science file
            ad = gt.convert_to_cal_header(adinput=ad, caltype="fringe", 
                                          keyword_comments=self.keyword_comments)[0]
            
            # Adding a PROCFRNG time stamp to the PHU
            gt.mark_history(adinput=ad, primname=self.myself(), keyword="PROCFRNG")
            
            # Refresh the AD types to reflect new processed status
            ad.refresh_types()
            
            adoutput_list.append(ad)
        
        # Upload to cal system
        rc.run("storeCalibration")
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc

    def separateLampOff(self, adinputs=None, stream='main', **params):
        """
        This primitive is intended to run on gcal imaging flats. 
        It goes through the input list and figures gout which ones are lamp-on
        and which ones are lamp-off. It can also cope with domeflats if their
        type is specified in the header keyword OBJECT.
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "separateLampOff", "starting"))

        # Initialize the list of output AstroData objects
        lampon_list = []
        lampoff_list = []

        # Loop over the input frames
        for ad in rc.get_inputs_as_astrodata():
            if('GCAL_IR_ON' in ad.types):
                log.stdinfo("%s is a lamp-on flat" % ad.data_label())
                lampon_list.append(ad)
            elif('GCAL_IR_OFF' in ad.types):
                log.stdinfo("%s is a lamp-off flat" % ad.data_label())
                lampoff_list.append(ad)
            elif('Domeflat OFF' in ad.phu_get_key_value('OBJECT')):
                log.stdinfo("%s is a lamp-off domeflat" % ad.data_label())
                lampoff_list.append(ad)
            elif('Domeflat' in ad.phu_get_key_value('OBJECT')):
                log.stdinfo("%s is a lamp-on domeflat" % ad.data_label())
                lampon_list.append(ad)                
            else:
                log.warning("Not a GCAL flatfield? Cannot tell if it is lamp-on or lamp-off for %s" % ad.data_label())

        rc.report_output(lampon_list, stream="lampOn")
        rc.report_output(lampoff_list, stream="lampOff")

        yield rc

    def separateFlatsDarks(self, adinputs=None, stream='main', **params):
        # Instantiate the log
        log = logutils.get_logger(__name__)

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "separateFlatsDarks", "starting"))

        # Initialize the list of AstroData objects to be added to the streams
        dark_list = []
        flat_list = []

        # Loop over the input frames
        for ad in rc.get_inputs_as_astrodata():
            all_types = "+".join(ad.types)
            if "DARK" in all_types:
                    dark_list.append(ad)
                    log.stdinfo("Dark : %s, %s" % (ad.data_label(), 
                                                        ad.filename))
            elif "FLAT" in all_types:
                    flat_list.append(ad)
                    log.stdinfo("Flat : %s, %s" % (ad.data_label(), 
                                                        ad.filename))
            else:
                log.warning("Not a Dark or a Flat : %s %s" % (
                                       ad.data_label(), ad.filename))
        if dark_list == []:
            log.warning("No Darks in input list")
        if flat_list == []:
            log.warning("No Flats in input list")

        rc.report_output(flat_list, stream="flats")
        rc.report_output(dark_list, stream="darks")

        yield rc

    def stackDarks(self, adinputs=None, stream='main', **params):
        # Instantiate the log
        log = logutils.get_logger(__name__)

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "stackDarks", "starting"))

        # Check darks for equal exposure time
        dark_list = rc.get_stream(stream="darks", style="AD")
        first_exp = dark_list[0].exposure_time().as_pytype()
        for ad in dark_list[1:]:
            file_exp = ad[0].exposure_time().as_pytype()
            if first_exp != file_exp:
                raise IOError("DARKS ARE NOT OF EQUAL EXPTIME")

        # stack the darks stream
        rc.run("showInputs(stream=darks)")
        rc.run("stackFrames(stream=darks)")

        yield rc

    def stackLampOnLampOff(self, adinputs=None, stream='main', **params):
        """
        This primitive stacks the Lamp On flats and the LampOff flats, then subtracts the two stacks
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "stackLampOnLampOff", "starting"))

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Get the lamp on list, stack it, and add the stack to the lampOnStack stream
        rc.run("showInputs(stream=lampOn)")
        rc.run("stackFrames(stream=lampOn)")

        # Get the lamp off list, stack it, and add the stack to the lampOnStack stream
        rc.run("showInputs(stream=lampOff)")
        rc.run("stackFrames(stream=lampOff)")

        yield rc

    def subtractLampOnLampOff(self, adinputs=None, stream='main', **params):
        """
        This primitive subtracts the lamp off stack from the lampon stack. It expects there to be only
        one file (the stack) on each stream - call stackLampOnLampOff to do the stacking before calling this
        """

        # Instantiate the log
        log = logutils.get_logger(__name__)

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "subtractLampOnLampOff", "starting"))

        # Initialize the list of output AstroData objects
        adoutput_list = []

        lampon_list = rc.get_stream(stream="lampOn", style="AD")
        lampoff_list = rc.get_stream(stream="lampOff", style="AD")

        if((len(lampon_list) > 0) and (len(lampoff_list) > 0)):
            lampon = lampon_list[0]
            lampoff = lampoff_list[0]
 
            log.stdinfo("Lamp ON is: %s %s" % (lampon.data_label(), lampon.filename))
            log.stdinfo("Lamp OFF is: %s %s" % (lampoff.data_label(), lampoff.filename))
            lampon.sub(lampoff)
            lampon.filename = gt.filename_updater(adinput=lampon, suffix="lampOnOff")

            adoutput_list.append(lampon)

        else:
            log.warning("Cannot subtract lamp on - lamp off flats as do not have some of each")
            if len(lampon_list) > 0:
                log.warning("Returning stacked lamp on flats")
                adoutput_list.extend(lampon_list)
            elif len(lampoff_list) > 0:
                log.warning("Returning stacked lamp off flats")
                adoutput_list.extend(lampoff_list)
            else:
                log.warning("Something is not quite right, no flats were accessible.")

        rc.report_output(adoutput_list)
        yield rc

    def makeBPM(self, adinputs=None, stream='main', **params):
        """
        To be run from recipe makeProcessedBPM.NIRI.

        Input is a stacked short darks and flats On/Off (1 filter)
        The flats are stacked and subtracted(ON - OFF)
        The Dark is stack of short darks
        """

        # avoid loading when BPM not required (lazy load)
        import numpy.ma as ma

        # instantiate variables - set limits using niflat defaults
        # used to exclude hot pixels from stddev calculation
        DARK_CLIP_THRESH = 5.0

        # Threshold above/below the N_SIGMA clipped median
        FLAT_HI_THRESH = 1.2
        FLAT_LO_THRESH = 0.8
        N_SIGMA = 3.0

        # Instantiate the log
        log = logutils.get_logger(__name__)

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "makeBPM", "starting"))

        # Get the subtracted, stacked flat from the on lampOn stream
        flat_stack = rc.get_stream(stream="lampOn", style="AD")

        if not flat_stack:
            raise IOError("A SET OF FLATS IS REQUIRED INPUT")
        else:
            flat = flat_stack[0]

        # Get the stacked dark on the dark stream
        dark_stack = rc.get_stream(stream="darks", style="AD")
        if not dark_stack:
            raise IOError("A SET OF DARKS IS REQUIRED INPUT")
        else:
            dark = dark_stack[0]

        # mask pixels of clipped mean (3 sigma) with a threshold.
        mean = ma.mean(flat.data)
        stddev = ma.std(flat.data)
        upper_lim = mean + stddev * N_SIGMA
        lower_lim = mean - stddev * N_SIGMA

        clipped_median = ma.median(ma.masked_outside(flat.data,
                                   lower_lim, upper_lim))

        upper_lim = FLAT_HI_THRESH * clipped_median
        lower_lim = FLAT_LO_THRESH * clipped_median

        log.stdinfo("BPM Flat Mask Lower < > Upper Limit: %s < > %s " % (
                                                    lower_lim, upper_lim))

        # mask Flats -- cold pixels
        flat_mask = ma.masked_outside(flat.data, lower_lim, upper_lim)

        # mask pixels outside 3 sigma * clipped standard deviation of median
        mean = ma.mean(dark.data)
        upper_lim = mean + (DARK_CLIP_THRESH * mean)
        lower_lim = mean - (DARK_CLIP_THRESH * mean)
        stddev = ma.std(ma.masked_outside(dark.data, lower_lim, upper_lim))

        clipped_median = ma.median(ma.masked_outside(dark.data, lower_lim, 
                                                     upper_lim))

        upper_lim = clipped_median + (N_SIGMA * stddev)
        lower_lim = clipped_median - (N_SIGMA * stddev)

        log.stdinfo("BPM Dark Mask Lower < > Upper Limit: %s < > %s " % (
                                                    lower_lim, upper_lim))

        # create the mask -- darks (hot pixels)
        dark_mask = ma.masked_outside(dark.data, upper_lim, lower_lim)

        # combine masks and write to bpm file
        data_mask = ma.mask_or(dark_mask.mask, flat_mask.mask)
        flat.data = data_mask.astype(float)

        flat.filename = gt.filename_updater(adinput=flat, suffix="_bpm")
        flat.set_key_value('OBJECT', 'BPM')
        flat.set_key_value('EXTNAME', 'DQ')
        rc.report_output(flat)

        yield rc

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
