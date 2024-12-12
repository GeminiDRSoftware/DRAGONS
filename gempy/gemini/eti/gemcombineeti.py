from copy import copy

from pyraf import iraf
from iraf import gemini
#from iraf import gemtools

from gempy.utils import logutils
from gempy.eti_core.pyrafeti import PyrafETI

from .gemcombinefile import InAtList, OutFile, LogFile
from .gemcombineparam import FlVardq, FlDqprop, Combine, \
    Masktype, Nlow, Nhigh, Reject, hardcoded_params, GemcombineParam

log = logutils.get_logger(__name__)

class GemcombineETI(PyrafETI):
    """This class coordinates the external task interface as it relates
    directly to the IRAF task: gemcombine
    """
    clparam_dict = None
    def __init__(self, inputs, params):
        """
        Adds the file and parameter objects to a list

        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("GemcombineETI __init__")
        PyrafETI.__init__(self, inputs, params)
        self.clparam_dict = {}
        self.add_file(InAtList(inputs, params))
        self.add_file(OutFile(inputs, params))
        self.add_file(LogFile(inputs, params))
        self.add_param(FlVardq(inputs, params))
        self.add_param(FlDqprop(inputs, params))
        self.add_param(Combine(inputs, params))
        self.add_param(Masktype(inputs, params))
        self.add_param(Nlow(inputs, params))
        self.add_param(Nhigh(inputs, params))
        self.add_param(Reject(inputs, params))
        for param in hardcoded_params:
            self.add_param(GemcombineParam(inputs, params, param,
                                           hardcoded_params[param]))

    def execute(self):
        """Execute pyraf task: gemcombine"""
        log.debug("GemcombineETI.execute()")

        # Populate object lists
        xcldict = copy(self.clparam_dict)
        for fil in self.file_objs:
            xcldict.update(fil.get_parameter())
        for par in self.param_objs:
            xcldict.update(par.get_parameter())
        iraf.unlearn(iraf.gemcombine)

        # Use setParam to list the parameters in the logfile
        for par in xcldict:
            #Stderr and Stdout are not recognized by setParam
            if par != "Stderr" and par != "Stdout":
                gemini.gemcombine.setParam(par, xcldict[par])
        log.fullinfo("\nGEMCOMBINE PARAMETERS:\n")
        iraf.lpar(iraf.gemcombine, Stderr=xcldict["Stderr"], \
            Stdout=xcldict["Stdout"])

        # Execute the task using the same dict as setParam
        # (but this time with Stderr and Stdout)
        try:
            gemini.gemcombine(**xcldict)
        except:
            # catch hard crash of the primitive
            raise RuntimeError("The IRAF task gemcombine failed")
        if gemini.gemcombine.status:
            # catch graceful exit on error
            raise RuntimeError("The IRAF task gemcombine failed")
        else:
            log.fullinfo("The IRAF task gemcombine completed successfully")

    def run(self):
        """Convenience function that runs all the needed operations."""
        log.debug("GemcombineETI.run()")
        self.prepare()
        self.execute()
        ad = self.recover()
        self.clean()
        return ad

    def recover(self):
        """Recovers reduction information into memory"""
        log.debug("GemcombineETI.recover()")
        ad = None
        for par in self.param_objs:
            par.recover()
        for fil in self.file_objs:
            if isinstance(fil, OutFile):
                ad = fil.recover()
            else:
                fil.recover()
        return ad
