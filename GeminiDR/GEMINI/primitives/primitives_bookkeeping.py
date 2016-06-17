# Prototype demo
import os
import time
import shutil

from datetime import datetime

from astrodata.utils import Errors
from astrodata.utils import logutils

from gempy.gemini import gemini_tools as gt

from recipe_system.reduction import IDFactory

from primitives_CORE import PrimitivesCORE

# log.stdinfo("PRIMITVE {} BEGIN".format(self.myself()))
# log.stdinfo("PRIMITVE {} END".format(self.myself()))

# ------------------------------------------------------------------------------
class Bookkeeping(PrimitivesCORE):
    """
    This is the class containing all of the bookkeeping primitives
    for the GEMINI level of the type hierarchy tree. It inherits all
    the primitives from the level above, 'COREPrimitives'.
    """
    tag = "GEMINI"

    def addToList(self, adinputs=None, stream='main', **params):
        """
        This primitive will update the lists of files to be stacked
        that have the same observationID with the current inputs.
        This file is cached between calls to reduce, thus allowing
        for one-file-at-a-time processing.
        
        :param purpose: 
        :type purpose: string
        
        """
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.addToList
        sfx = p_pars["suffix"]
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))

        timestamp_key = self.timestamp_keys[self.myself()]
        if adinputs:
            self.adinputs = adinputs

        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(p_pars))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        # Add the appropriate time stamps to the PHU
        gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        logutils.update_indent(0)
        return


    def getList(self, adinputs=None, stream='main', **params):
        """
        This primitive will check the files in the stack lists are on disk,
        and then update the inputs list to include all members of the stack 
        for stacking.
        
        :param purpose: 
        :type purpose: string
        """
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.getList
        sfx = p_pars["suffix"]
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))

        timestamp_key = self.timestamp_keys[self.myself()]
        if adinputs:
            self.adinputs = adinputs

        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(p_pars))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        # Add the appropriate time stamps to the PHU
        gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        logutils.update_indent(0)
        return


    def showList(self, adinputs=None, stream='main', **params):
        """
        This primitive will log the list of files in the stacking list matching
        the current inputs and 'purpose' value.
        
        :param purpose: 
        :type purpose: string
        """
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.showList
        sfx = p_pars["suffix"]
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))

        timestamp_key = self.timestamp_keys[self.myself()]
        if adinputs:
            self.adinputs = adinputs

        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(p_pars))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        # Add the appropriate time stamps to the PHU
        gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        logutils.update_indent(0)
        return

    def writeOutputs(self, adinputs=None, stream='main', **params):
        """
        A primitive that may be called by a recipe at any stage to
        write the outputs to disk.
        If suffix is set during the call to writeOutputs, any previous 
        suffixs will be striped and replaced by the one provided.
        examples: 
        writeOutputs(suffix= '_string'), writeOutputs(prefix= '_string') 
        or if you have a full file name in mind for a SINGLE file being 
        ran through Reduce you may use writeOutputs(outfilename='name.fits').
        
        :param strip: Strip the previously suffixed strings off file name?
        :type strip: Python boolean (True/False)
                     default: False
        
        :param clobber: Write over any previous file with the same name that
                        all ready exists?
        :type clobber: Python boolean (True/False)
                       default: False
        
        :param suffix: Value to be post pended onto each input name(s) to 
                       create the output name(s).
        :type suffix: string
        
        :param prefix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type prefix: string
        
        :param outfilename: The full filename you wish the file to be written
                            to. Note: this only works if there is ONLY ONE file
                            in the inputs.
        :type outfilename: string

        """
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.writeOutputs
        sfx = p_pars["suffix"]
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))

        timestamp_key = self.timestamp_keys[self.myself()]
        if adinputs:
            self.adinputs = adinputs

        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(p_pars))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        # Add the appropriate time stamps to the PHU
        gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        logutils.update_indent(0)
        return
