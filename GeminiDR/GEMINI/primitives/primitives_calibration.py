from astrodata.utils import logutils
from gempy.gemini import gemini_tools as gt

from primitives_CORE import PrimitivesCORE

# ------------------------------------------------------------------------------
class Calibration(PrimitivesCORE):
    tag = "GEMINI"

    def failCalibration(self, adinputs=None, stream='main', **params):
        # Mark a given calibration "fail" and upload it 
        # to the system. This is intended to be used to mark a 
        # calibration file that has already been uploaded, so that
        # it will not be returned as a valid match for future data.
        
        # Instantiate the log

        return

    def getCalibration(self, adinputs=None, stream='main', **params):
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.getCalibration
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))

        if adinputs:
            self.adinputs = adinputs

        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(p_pars))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        logutils.update_indent(0)
        return

    def getProcessedArc(self, adinputs=None, stream='main', **params):
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = {'caltype':'processed_arc'}
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))

        if adinputs:
            self.adinputs = adinputs

        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(p_pars))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        self.getCalibration(caltype="processed_arc")
        logutils.update_indent(0)
        return
    
    def getProcessedBias(self, adinputs=None, stream='main', **params):
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = {'caltype':'processed_bias'}
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))

        if adinputs:
            self.adinputs = adinputs

        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(p_pars))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        self.getCalibration(caltype="processed_bias")
        logutils.update_indent(0)
        return


    def getProcessedDark(self, adinputs=None, stream='main', **params):
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = {'caltype':'processed_dark'}
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))

        if adinputs:
            self.adinputs = adinputs

        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(p_pars))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        self.getCalibration(caltype="processed_dark")
        logutils.update_indent(0)
        return
    
    def getProcessedFlat(self, adinputs=None, stream='main', **params):
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = {'caltype':'processed_flat'}
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))

        if adinputs:
            self.adinputs = adinputs

        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(p_pars))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        self.getCalibration(caltype="processed_flat")
        logutils.update_indent(0)
        return
    
    def getProcessedFringe(self, adinputs=None, stream='main', **params):
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = {'caltype':'processed_fringe'}
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))

        if adinputs:
            self.adinputs = adinputs

        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(p_pars))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        self.getCalibration(caltype="processed_fringe")
        logutils.update_indent(0)
        return
    
    def storeCalibration(self, adinputs=None, stream='main', **params):
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = {'action':'store'}
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))

        if adinputs:
            self.adinputs = adinputs

        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(p_pars))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        logutils.update_indent(0)
        return
    

    def storeProcessedArc(self, adinputs=None, stream='main', **params):
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.storeProcessedArc
        sfx = p_pars["suffix"]
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))

        if adinputs:
            self.adinputs = adinputs

        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(p_pars))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        # Add the appropriate time stamps to the PHU

        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        self.storeCalibration()
        logutils.update_indent(0)
        return
    
    def storeProcessedBias(self, adinputs=None, stream='main', **params):
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.storeProcessedBias
        sfx = p_pars["suffix"]
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))

        if adinputs:
            self.adinputs = adinputs

        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(p_pars))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        # Add the appropriate time stamps to the PHU

        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        self.storeCalibration()
        logutils.update_indent(0)
        return

    def storeBPM(self, adinputs=None, stream='main', **params):
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.storeBPM
        sfx = p_pars["suffix"]
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))

        if adinputs:
            self.adinputs = adinputs

        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(p_pars))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        # Add the appropriate time stamps to the PHU

        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        self.storeCalibration()
        logutils.update_indent(0)
        return

    def storeProcessedDark(self, adinputs=None, stream='main', **params):
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.storeProcessedDark
        sfx = p_pars["suffix"]
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))

        if adinputs:
            self.adinputs = adinputs

        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(p_pars))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        # Add the appropriate time stamps to the PHU

        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        self.storeCalibration()
        logutils.update_indent(0)
        return
    
    def storeProcessedFlat(self, adinputs=None, stream='main', **params):
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.storeProcessedFlat
        sfx = p_pars["suffix"]
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))

        if adinputs:
            self.adinputs = adinputs

        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(p_pars))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        # Add the appropriate time stamps to the PHU

        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        self.storeCalibration()
        logutils.update_indent(0)
        return
    
    def storeProcessedFringe(self, adinputs=None, stream='main', **params):
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.storeProcessedFringe
        sfx = p_pars["suffix"]
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))

        if adinputs:
            self.adinputs = adinputs

        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(p_pars))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        # Add the appropriate time stamps to the PHU
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        self.storeCalibration()
        logutils.update_indent(0)

        return
