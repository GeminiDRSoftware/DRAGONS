import sys
from copy import copy

from pyraf import iraf
from iraf import gemini
from iraf import gemtools

from astrodata.utils import Errors
from astrodata.utils import logutils
from astrodata.utils.gemutil import pyrafLoader
from astrodata.eti.pyrafeti import PyrafETI

from gemcombinefile import InAtList, OutFile, LogFile
from gemcombineparam import FlVardq, FlDqprop, Combine, \
    Masktype, Nlow, Nhigh, Reject, hardcoded_params, GemcombineParam
    
log = logutils.get_logger(__name__)

class GemcombineETI(PyrafETI):
    """This class coordinates the external task interface as it relates
    directly to the IRAF task: gemcombine
    """
    clparam_dict = None
    def __init__(self, rc):
        """
        Adds the file and parameter objects to a list

        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("GemcombineETI __init__")
        PyrafETI.__init__(self, rc)
        self.clparam_dict = {}
        self.add_file(InAtList(rc))
        self.add_file(OutFile(rc))
        self.add_file(LogFile(rc))
        self.add_param(FlVardq(rc))
        self.add_param(FlDqprop(rc))
        self.add_param(Combine(rc))
        self.add_param(Masktype(rc))
        self.add_param(Nlow(rc))
        self.add_param(Nhigh(rc))
        self.add_param(Reject(rc))
        for param in hardcoded_params:
            self.add_param(GemcombineParam(rc, param, hardcoded_params[param]))

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
            if par != "Stderr" and par !="Stdout":
                gemini.gemcombine.setParam(par,xcldict[par])
        log.fullinfo("\nGEMCOMBINE PARAMETERS:\n")
        iraf.lpar(iraf.gemcombine, Stderr=xcldict["Stderr"], \
            Stdout=xcldict["Stdout"])

        # Execute the task using the same dict as setParam
        # (but this time with Stderr and Stdout)
        try:
            gemini.gemcombine(**xcldict)
        except:
            # catch hard crash of the primitive
            raise Errors.OutputError("The IRAF task gemcombine failed")
        if gemini.gemcombine.status:
            # catch graceful exit on error
            raise Errors.OutputError("The IRAF task gemcombine failed")
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
        
        
