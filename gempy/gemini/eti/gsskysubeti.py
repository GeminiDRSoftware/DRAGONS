import sys
from copy import copy

from pyraf import iraf
from iraf import gemini
from iraf import gmos
from iraf import gemtools

from astrodata.utils import Errors
from astrodata.utils import logutils
from astrodata.utils.gemutil import pyrafLoader
from astrodata.eti.pyrafeti import PyrafETI
from gsskysubfile import InAtList, OutAtList, LogFile
from gsskysubparam import FlVardq, hardcoded_params, GsskysubParam
    
log = logutils.get_logger(__name__)

class GsskysubETI(PyrafETI):
    """This class coordinates the external task interface as it relates
    directly to the IRAF task: gsskysub
    """
    clparam_dict = None
    ad = None
    def __init__(self, rc, ad):
        """
        Adds the file and parameter objects to a list

        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("GsskysubETI __init__")
        PyrafETI.__init__(self, rc)
        self.clparam_dict = {}

        # if ad then it will only process the ad
        self.add_file(InAtList(rc, ad))
        self.add_file(OutAtList(rc, ad))
        self.add_file(LogFile(rc))
        self.add_param(FlVardq(rc, ad))
        for param in hardcoded_params:
            self.add_param(GsskysubParam(rc, param,
                                         hardcoded_params[param]))

    def execute(self):
        """Execute pyraf task: gsskysub"""
        log.debug("GsskysubETI.execute()")

        # Populate object lists
        xcldict = copy(self.clparam_dict)
        for fil in self.file_objs:
            xcldict.update(fil.get_parameter())
        for par in self.param_objs:
            xcldict.update(par.get_parameter())
        iraf.unlearn(iraf.gmos.gsskysub)

        # Use setParam to list the parameters in the logfile 
        for par in xcldict:
            #Stderr and Stdout are not recognized by setParam
            if par != "Stderr" and par !="Stdout":
                gemini.gmos.gsskysub.setParam(par,xcldict[par])
        log.fullinfo("\nGSSKYSUB PARAMETERS:\n")
        iraf.lpar(iraf.gmos.gsskysub, Stderr=xcldict["Stderr"],
                  Stdout=xcldict["Stdout"])

        # Execute the task using the same dict as setParam
        # (but this time with Stderr and Stdout) 
        #from pprint import pprint
        #pprint(xcldict)
        gemini.gmos.gsskysub(**xcldict)
        if gemini.gmos.gsskysub.status:
            raise Errors.OutputError("The IRAF task gmos.gsskysub failed")
        else:
            log.fullinfo("The IRAF task gmos.gsskysub completed successfully")

    def run(self):
        """Convenience function that runs all the needed operations."""
        log.debug("GsskysubETI.run()")
        adlist = []
        self.prepare()
        self.execute()
        adlist = self.recover()
        self.clean()
        return adlist

    def recover(self):
        """Recovers reduction information into memory"""
        log.debug("GsskysubETI.recover()")
        adlist = []
        for par in self.param_objs:
            par.recover()
        for fil in self.file_objs:
            if isinstance(fil, OutAtList):
                adlist.extend(fil.recover())
            else:
                fil.recover()
        if len(adlist) == 1:
            return adlist[0]
        else:
            return adlist
        
        
