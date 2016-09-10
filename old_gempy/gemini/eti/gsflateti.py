import sys
from copy import copy

from pyraf import iraf
from iraf import gemini
from iraf import gemtools

from astrodata.utils import Errors
from astrodata.utils import logutils
from astrodata.utils.gemutil import pyrafLoader
from astrodata.eti.pyrafeti import PyrafETI
from gsflatfile import InAtList, OutFile, LogFile
from gsflatparam import FlVardq, hardcoded_params, GsflatParam

log = logutils.get_logger(__name__)

class GsflatETI(PyrafETI):
    """This class coordinates the external task interface as it relates
    directly to the IRAF task: gsflat
    """
    clparam_dict = None
    def __init__(self, rc):
        """
        Adds the file and parameter objects to a list

        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("GsflatETI __init__")
        PyrafETI.__init__(self, rc)
        self.clparam_dict = {}
        self.add_file(InAtList(rc))
        self.add_file(OutFile(rc))
        self.add_file(LogFile(rc))
        self.add_param(FlVardq(rc))
        for param in hardcoded_params:
            self.add_param(GsflatParam(rc, param, hardcoded_params[param]))

    def execute(self):
        """Execute pyraf task: gsflat"""
        log.debug("GsflatETI.execute()")

        # Populate object lists
        xcldict = copy(self.clparam_dict)
        for fil in self.file_objs:
            xcldict.update(fil.get_parameter())
        for par in self.param_objs:
            xcldict.update(par.get_parameter())
        iraf.unlearn(iraf.gsflat)

        # Use setParam to list the parameters in the logfile 
        for par in xcldict:
            #Stderr and Stdout are not recognized by setParam
            if par != "Stderr" and par !="Stdout":
                gemini.gsflat.setParam(par,xcldict[par])
        log.fullinfo("\nGSFLAT PARAMETERS:\n")
        iraf.lpar(iraf.gsflat, Stderr=xcldict["Stderr"], \
            Stdout=xcldict["Stdout"])

        # Execute the task using the same dict as setParam
        # (but this time with Stderr and Stdout) 
        gemini.gsflat(**xcldict)
        if gemini.gsflat.status:
            raise Errors.OutputError("The IRAF task gsflat failed")
        else:
            log.fullinfo("The IRAF task gsflat completed successfully")

    def run(self):
        """Convenience function that runs all the needed operations."""
        log.debug("GsflatETI.run()")
        self.prepare()
        self.execute()
        ad = self.recover()
        self.clean()
        return ad

    def recover(self):
        """Recovers reduction information into memory"""
        log.debug("GsflatETI.recover()")
        ad = None
        for par in self.param_objs:
            par.recover()
        for fil in self.file_objs:
            if isinstance(fil, OutFile):
                ad = fil.recover()
            else:
                fil.recover()
        return ad
        
        
