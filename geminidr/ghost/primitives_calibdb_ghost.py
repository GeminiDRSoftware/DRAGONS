#
#                                                                  gemini_python
#
#                                                    primitives_calibdb_ghost.py
# ------------------------------------------------------------------------------
from gempy.gemini import gemini_tools as gt

from geminidr.core import CalibDB
from . import parameters_calibdb_ghost


from recipe_system.utils.decorators import parameter_override

# ------------------------------------------------------------------------------
@parameter_override
class CalibDBGHOST(CalibDB):
    """
    This is the class containing all of the calibration bookkeeping primitives
    for the CalibDBGHOST level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = set()  # Not allowed to be a selected as a primitivesClass

    def __init__(self, adinputs, **kwargs):
        super(CalibDBGHOST, self).__init__(adinputs, **kwargs)
        self.inst_lookups = 'geminidr.ghost.lookups'
        self._param_update(parameters_calibdb_ghost)

    #def getProcessedArc(self, adinputs=None, **params):
    #    procmode = 'sq' if self.mode == 'sq' else None
    #    cals = self.caldb.get_processed_arc(adinputs, procmode=procmode)
    #    self._assert_calibrations(adinputs, cals)
    #    return adinputs

    def getProcessedSlit(self, adinputs=None, **params):
        procmode = 'sq' if self.mode == 'sq' else None
        cals = self.caldb.get_processed_slit(adinputs, procmode=procmode)
        self._assert_calibrations(adinputs, cals)
        return adinputs

    def getProcessedSlitFlat(self, adinputs=None, **params):
        procmode = 'sq' if self.mode == 'sq' else None
        cals = self.caldb.get_processed_slitflat(adinputs, procmode=procmode)
        self._assert_calibrations(adinputs, cals)
        return adinputs

    # =========================== STORE PRIMITIVES =================================
    def storeProcessedSlit(self, adinputs=None, suffix=None, force=False):
        caltype = 'processed_slit'
        self.log.debug(gt.log_message("primitive", self.myself(), "starting"))
        adinputs = self._markAsCalibration(adinputs, suffix=suffix,
                                    primname=self.myself(), keyword="PRSLITIM")
        self.storeCalibration(adinputs, caltype=caltype)
        return adinputs

    def storeProcessedSlitFlat(self, adinputs=None, suffix=None, force=False):
        caltype = 'processed_slitflat'
        self.log.debug(gt.log_message("primitive", self.myself(), "starting"))
        adinputs = self._markAsCalibration(adinputs, suffix=suffix,
                                    primname=self.myself(), keyword="PRSLITFL")
        self.storeCalibration(adinputs, caltype=caltype)
        return adinputs

    def storeProcessedStandard(self, adinputs=None, suffix=None):
        caltype = 'processed_standard'
        self.log.debug(gt.log_message("primitive", self.myself(), "starting"))
        adinputs = self._markAsCalibration(
            adinputs, suffix=suffix, primname=self.myself(), keyword="PROCSTND")
        self.storeCalibration(adinputs, caltype=caltype)
        return adinputs

