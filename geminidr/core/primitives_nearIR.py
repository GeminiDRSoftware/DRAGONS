#
#                                                                  gemini_python
#
#                                                           primitives_nearIR.py
# ------------------------------------------------------------------------------
import os
import numpy.ma as ma

import astrodata
import gemini_instruments

from gempy.gemini import gemini_tools as gt

from recipe_system.utils.decorators import parameter_override

from geminidr.core.parameters_NIR import ParametersNIR

from geminidr import PrimitivesBASE
# ------------------------------------------------------------------------------
# NOTE: This module/class is not complete. 
# Primitives here were transferred from primitives_calibdb:
# - makeBPM
# - separateLampOff
# - separateFlatsDarks
# - stackDarks
# - stackLampOnLampOff
# - subtractLampOnLampOff
#
# The NIR class does *not* yetinclude other primitives specified in the 
# Refactoring Coordination document:
# - lampOnLmapOff
# - makeLampOffFlat
# - makeLampOnLampOffFlat
# - thermalEmissionCorrect
# (kra 09-12-16)
# ------------------------------------------------------------------------------
@parameter_override
class NIR(PrimitivesBASE):
    tagset = None

    def __init__(self, adinputs, **kwargs):
        super(NIR, self).__init__(adinputs, **kwargs)
        self.parameters = ParametersNIR

    def makeBPM(self, adinputs=None, stream='main', **params):
        """
        To be run from recipe makeProcessedBPM

        Input is a stacked short darks and flats On/Off (1 filter)
        The flats are stacked and subtracted (ON - OFF)
        The Dark is stack of short darks

        """

        # To exclude hot pixels from stddev calculation
        DARK_CLIP_THRESH = 5.0

        # Threshold above/below the N_SIGMA clipped median
        FLAT_HI_THRESH = 1.2
        FLAT_LO_THRESH = 0.8
        N_SIGMA = 3.0

        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        # Get the subtracted, stacked flat from the on lampOn stream
        flat_stack = self.streams.get("lampOn")
        if not flat_stack:
            raise IOError("A SET OF FLATS IS REQUIRED INPUT")
        else:
            flat = flat_stack[0]

        # Get the stacked dark on the dark stream
        dark_stack = self.streams.get("darks")
        if not dark_stack:
            raise IOError("A SET OF DARKS IS REQUIRED INPUT")
        else:
            dark = dark_stack[0]

        # mask pixels of clipped mean (3 sigma) with a threshold.
        mean = ma.mean(flat.data)
        stddev = ma.std(flat.data)
        upper_lim = mean + stddev * N_SIGMA
        lower_lim = mean - stddev * N_SIGMA

        clipped_median = ma.median(ma.masked_outside(flat.data, lower_lim, upper_lim))

        upper_lim = FLAT_HI_THRESH * clipped_median
        lower_lim = FLAT_LO_THRESH * clipped_median

        msg = "BPM Flat Mask Lower < > Upper Limit: {} < > {} " 
        log.stdinfo(msg.format(lower_lim, upper_lim))

        # mask Flats -- cold pixels
        flat_mask = ma.masked_outside(flat.data, lower_lim, upper_lim)

        # mask pixels outside 3 sigma * clipped standard deviation of median
        mean = ma.mean(dark.data)
        upper_lim = mean + (DARK_CLIP_THRESH * mean)
        lower_lim = mean - (DARK_CLIP_THRESH * mean)
        stddev = ma.std(ma.masked_outside(dark.data, lower_lim, upper_lim))

        clipped_median = ma.median(ma.masked_outside(dark.data, lower_lim, upper_lim))

        upper_lim = clipped_median + (N_SIGMA * stddev)
        lower_lim = clipped_median - (N_SIGMA * stddev)

        msg = "BPM Dark Mask Lower < > Upper Limit: {} < > {}"
        log.stdinfo(msg.format(lower_lim, upper_lim))

        # create the mask -- darks (hot pixels)
        dark_mask = ma.masked_outside(dark.data, upper_lim, lower_lim)

        # combine masks and write to bpm file
        data_mask = ma.mask_or(dark_mask.mask, flat_mask.mask)
        flat.data = data_mask.astype(float)

        flat.filename = gt.filename_updater(adinput=flat, suffix="_bpm")
        flat.set_key_value('OBJECT', 'BPM')
        flat.set_key_value('EXTNAME', 'DQ')

        # @TODO rc.report_output(flat) 

        return adinputs

    def separateLampOff(self, adinputs=None, stream='main', **params):
        """
        This primitive is intended to run on gcal imaging flats. 
        It goes through the input list and figures out which ones are lamp-on
        and which ones are lamp-off. It can also cope with domeflats if their
        type is specified in the header keyword OBJECT.

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        # Initialize the list of output AstroData objects
        lampon_list = []
        lampoff_list = []

        # Loop over the input frames
        for ad in adinputs:
            if('GCAL_IR_ON' in ad.tags):
                log.stdinfo("%s is a lamp-on flat" % ad.data_label())
                lampon_list.append(ad)
            elif('GCAL_IR_OFF' in ad.tags):
                log.stdinfo("%s is a lamp-off flat" % ad.data_label())
                lampoff_list.append(ad)
            elif('Domeflat OFF' in ad.phu_get_key_value('OBJECT')):
                log.stdinfo("%s is a lamp-off domeflat" % ad.data_label())
                lampoff_list.append(ad)
            elif('Domeflat' in ad.phu_get_key_value('OBJECT')):
                log.stdinfo("%s is a lamp-on domeflat" % ad.data_label())
                lampon_list.append(ad)                
            else:
                warn = "Not a GCAL flatfield? "
                warn += "Cannot determine lamp-on or lamp-off for {}"
                log.warning(warn.format(ad.data_label()))

        self.streams.update({"lampOn" : lampOn_list})
        self.streamd.update({"lampOff": lampOff_list})

        return adinputs

    def separateFlatsDarks(self, adinputs=None, stream='main', **params):
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        # Initialize lists of AstroData objects to be added to the streams
        dark_list = []
        flat_list = []
        for ad in adinputs:
            all_tags = "+".join(ad.tags)
            if "DARK" in all_tags:
                dark_list.append(ad)
                log.stdinfo("Dark: {}, {}".format(ad.data_label(), ad.filename))
            elif "FLAT" in all_tags:
                flat_list.append(ad)
                log.stdinfo("Flat: {}, {}".format(ad.data_label(), ad.filename))
            else:
                log.warning("Not Dark/Flat: {} {}".format(ad.data_label(), 
                                                          ad.filename))
        if not dark_list:
            log.warning("No Darks in input list")
        if not flat_list:
            log.warning("No Flats in input list")

        self.streams.update({"flats" : flat_list})
        self.streams.update({"darks" : dark_list})

        return adinputs

    def stackDarks(self, adinputs=None, stream='main', **params):
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        # Check darks for equal exposure time
        dark_list = self.streams.get("darks")
        first_exp = dark_list[0].exposure_time()
        for ad in dark_list[1:]:
            file_exp = ad[0].exposure_time()
            if first_exp != file_exp:
                raise IOError("DARKS ARE NOT OF EQUAL EXPTIME")

        # stack the darks stream
        rc.run("showInputs(stream=darks)")
        rc.run("stackFrames(stream=darks)")

        yield rc

    def stackLampOnLampOff(self, adinputs=None, stream='main', **params):
        """
        This primitive stacks the Lamp On flats and the LampOff flats, 
        then subtracts the two stacks.

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

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

