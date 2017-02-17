#
#                                                                  gemini_python
#
#                                                           primitives_nearIR.py
# ------------------------------------------------------------------------------
import numpy as np
from astropy.stats import sigma_clip

from gempy.gemini import gemini_tools as gt

from geminidr import PrimitivesBASE
from .parameters_nearIR import ParametersNearIR
from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------

@parameter_override
class NearIR(PrimitivesBASE):
    tagset = None

    def __init__(self, adinputs, **kwargs):
        super(NearIR, self).__init__(adinputs, **kwargs)
        self.parameters = ParametersNearIR

    def makeBPM(self, adinputs=None, **params):
        """
        To be run from recipe makeProcessedBPM

        Input is a stacked short darks and flats On/Off (1 filter)
        The flats are stacked and subtracted (ON - OFF)
        The Dark is stack of short darks
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        # To exclude hot pixels from stddev calculation
        DARK_CLIP_THRESH = 5.0

        # Threshold above/below the N_SIGMA clipped median
        FLAT_HI_THRESH = 1.2
        FLAT_LO_THRESH = 0.8
        N_SIGMA = 3.0

        # Get the stacked flat and dark; these are single-element lists
        try:
            flat = self.streams['lampOn'][0]
        except (KeyError, TypeError):
            raise IOError("A SET OF FLATS IS REQUIRED INPUT")
        try:
            dark = self.streams['darks'][0]
        except (KeyError, TypeError):
            raise IOError("A SET OF DARKS IS REQUIRED INPUT")

        for dark_ext, flat_ext in zip(dark, flat):
            # Dubious clipping of the flat
            clipped_median= np.ma.median(sigma_clip(flat_ext.data, sigma=N_SIGMA,
                                                    iters=1, cenfunc=np.ma.mean))
            upper_lim = FLAT_HI_THRESH * clipped_median
            lower_lim = FLAT_LO_THRESH * clipped_median
            msg = "BPM Flat Mask Lower < > Upper Limit: {} < > {} "
            log.stdinfo(msg.format(lower_lim, upper_lim))
            flat_mask = np.ma.masked_outside(flat_ext.data, lower_lim, upper_lim)

            # Dubious clipping of the dark
            mean = np.mean(dark_ext.data)
            upper_lim = mean + (DARK_CLIP_THRESH * mean)
            lower_lim = mean - (DARK_CLIP_THRESH * mean)
            stddev = np.ma.std(np.ma.masked_outside(dark_ext.data,
                                                    lower_lim, upper_lim))
            clipped_median = np.ma.median(np.ma.masked_outside(dark_ext.data,
                                                        lower_lim, upper_lim))
            upper_lim = clipped_median + (N_SIGMA * stddev)
            lower_lim = clipped_median - (N_SIGMA * stddev)

            msg = "BPM Dark Mask Lower < > Upper Limit: {} < > {}"
            log.stdinfo(msg.format(lower_lim, upper_lim))

            # create the mask -- darks (hot pixels)
            dark_mask = np.ma.masked_outside(dark_ext.data, upper_lim, lower_lim)

            # combine masks and write to bpm file
            data_mask = np.ma.mask_or(dark_mask.mask, flat_mask.mask)
            flat_ext.reset(data_mask.astype(np.int16), mask=None, variance=None)

        flat.filename = gt.filename_updater(adinput=flat, suffix="_bpm")
        flat.phu.set('OBJECT', 'BPM')
        return [flat]

    def lampOnLampOff(self, adinputs=None, **params):
        """
        This separates the lamp-on and lamp-off flats, stacks them, and subtracts
        one from the other, and returns that single frame. It uses streams to
        propagate the frames, hence there's no need to send/collect adinputs.
        """
        self.separateLampOff(adinputs)
        self.stackLampOnLampOff(None)
        return self.subtractLampOnLampOff(None)

    def separateFlatsDarks(self, adinputs=None, **params):
        """
        This primitive produces two streams, one containing flats, and one
        containing darks
        """
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

    def separateLampOff(self, adinputs=None, **params):
        """
        This primitive is intended to run on gcal imaging flats.
        It goes through the input list and figures out which ones are lamp-on
        and which ones are lamp-off, and pushes these to two streams.
        It can also cope with domeflats if their type is specified in the
        header keyword OBJECT.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        # Initialize the list of output AstroData objects
        lampon_list = []
        lampoff_list = []

        for ad in adinputs:
            if 'GCAL_IR_ON' in ad.tags:
                log.stdinfo("{} is a lamp-on flat".format(ad.filename))
                lampon_list.append(ad)
            elif 'GCAL_IR_OFF' in ad.tags:
                log.stdinfo("{} is a lamp-off flat".format(ad.filename))
                lampoff_list.append(ad)
            elif ('Domeflat OFF' in ad.phu.get('OBJECT')):
                log.stdinfo("{} is a lamp-off domeflat".format(ad.filename))
                lampoff_list.append(ad)
            elif ('Domeflat' in ad.phu.get('OBJECT')):
                log.stdinfo("{} is a lamp-on domeflat".format(ad.filename))
                lampon_list.append(ad)
            else:
                log.warning("Cannot determine lamp-on/off for {}".
                            format(ad.filename))

        self.streams.update({"lampOn" : lampon_list, "lampOff": lampoff_list})
        return adinputs

    def stackDarks(self, adinputs=None, **params):
        """
        This primitive stacks the files in the "darks" stream, after checking
        they have the same exposure time, and returns the output there.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        # Check darks for equal exposure time
        try:
            dark_list = self.streams["darks"]
        except KeyError:
            log.warning("There are no darks in the 'darks' stream.")
            return adinputs

        if not all(dark.exposure_time()==dark_list[0].exposure_time()
                   for dark in dark_list[1:]):
                raise IOError("DARKS ARE NOT OF EQUAL EXPTIME")

        # stack the darks stream
        self.showInputs(stream='darks')
        self.streams['darks'] = self.stackFrames(self.streams['darks'])
        return adinputs

    def stackLampOnLampOff(self, adinputs=None, **params):
        """
        This primitive stacks the Lamp On flats and the LampOff flats from
        the streams, and returns the two stacked flats to the streams.
        It doesn't even look at adinputs.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        if self.streams.get('lampOn'):
            self.showInputs(stream='lampOn')
            self.streams['lampOn'] = self.stackFrames(self.streams['lampOn'])

        if self.streams.get('lampOff'):
            self.showInputs(stream='lampOff')
            self.streams['lampOff'] = self.stackFrames(self.streams['lampOff'])
        return adinputs

    def subtractLampOnLampOff(self, adinputs=None, **params):
        """
        This primitive subtracts the lamp off stack from the lampon stack.
        It expects there to be only one file (the stack) in each stream -
        call stackLampOnLampOff to do the stacking before calling this
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        lampon_list = self.streams.get('lampOn', [])
        lampoff_list = self.streams.get('lampOff', [])

        if lampon_list and lampoff_list:
            if len(lampon_list) > 1:
                log.warning('More than one file in the lampOn stream. '
                            'Using first file only.')
            if len(lampoff_list) > 1:
                log.warning('More than one file in the lampOff stream. '
                            'Using first file only.')
            lampon = lampon_list[0]
            lampoff = lampoff_list[0]
            log.stdinfo("Lamp ON is:  {}".format(lampon.filename))
            log.stdinfo("Lamp OFF is: {}".format(lampoff.filename))
            lampon.subtract(lampoff)
            lampon.filename = gt.filename_updater(lampon, suffix="lampOnOff")
            del self.streams['lampOn'], self.streams['lampOff']
            return [lampon]
        else:
            log.warning("Cannot subtract lamp on - lamp off flats as do not "
                        "have some of each")
            if len(lampon_list) > 0:
                log.warning("Returning stacked lamp on flats")
                return lampon_list
            elif len(lampoff_list) > 0:
                return []
            else:
                log.warning("Something is not right, no flats were accessible.")
                return []