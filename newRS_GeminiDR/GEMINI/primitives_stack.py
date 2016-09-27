# Prototype demo
from astrodata.utils import logutils
from gempy.gemini import gemini_tools as gt

from .parameters_stack import ParametersStack
from .primitives_CORE import PrimitivesCORE

from rpms.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class Stack(PrimitivesCORE):
    """
    This is the class containing all of the primitives for the GEMINI level of
    the type hierarchy tree. It inherits all the primitives from the level
    above, 'GENERALPrimitives'.

    """
    tagset = set(["GEMINI"])

    def __init__(self, adinputs, context, ucals=None, uparms=None):
        super(Stack, self).__init__(adinputs, context, ucals=ucals, uparms=uparms)
        self.parameters = ParametersStack
    
    def alignAndStack(self, adinputs=None, stream='main', **params):
        """
        This primitive calls a set of primitives to perform the steps
        needed for alignment of frames to a reference image and stacking.
        
        :param check_if_stack: Parameter to call a check as to whether 
                               stacking should be performed. If not, this
                               part of the recipe is skipped and the single
                               input file returned.
        :type check_if_stack: bool
        """
        
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.addToList
        sfx = ''
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

        self.addToList(purpose='forStack')
        self.getList(purpose='forStack')
        self.detectSources()
        self.correctWCSToReferenceFrame()
        self.alignToReferenceFrame()
        self.correctBackgroundToReferenceImage()
        self.stackFrames()

        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)


        return
        
    
    def stackFrames(self, adinputs=None, stream='main', **params):
        """
        This primitive will stack each science extension in the input dataset.
        New variance extensions are created from the stacked science extensions
        and the data quality extensions are propagated through to the final
        file.
        
        :param operation: type of combining operation to use.
        :type operation: string, options: 'average', 'median'.
        
        :param reject_method: type of rejection algorithm
        :type reject_method: string, options: 'avsigclip', 'minmax', None
        
        :param mask: Use the data quality extension to mask bad pixels?
        :type mask: bool
        
        :param nlow: number of low pixels to reject (used with
                     reject_method=minmax)
        :type nlow: int
        
        :param nhigh: number of high pixels to reject (used with
                      reject_method=minmax)
        :type nhigh: int
        
        """
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.stackFrames
        sfx = p_pars["suffix"]
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))

        if stream is not 'main':
            log.stdinfo("Received stream:{}".format(stream))

        if params:
            log.stdinfo("Received parameter set \n {}for stream:{}".format(params))
            log.stdinfo("Updating ...")
            #self._override_pars(params)

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

        return

    
    def stackSkyFrames(self, adinputs=None, stream='main', **params):
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.stackSkyFrames
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
    
        self.showInputs(stream='forStack')

        sfx    = p_pars['suffix']
        op     = p_pars['operation']
        mask   = p_pars['mask']
        low    = p_pars['nlow']
        high   = p_pars['nhigh']
        reject = p_pars['reject_method']

        self.stackFrames(stream ='forStack', suffix=sfx, operation=op, mask=mask,
                         nlow=low, nhigh=high, reject_method=reject)

        self.showInputs(stream='forStack')
                    
       # Add the appropriate time stamps to the PHU
        gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        self.showInputs(stream='forSkyCorrection')
        return
