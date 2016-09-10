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

from gireducefile import InAtList, OutAtList, LogFile
from gireduceparam import Nbiascontam, \
                        subtract_overscan_hardcoded_params, GireduceParam
    
log = logutils.get_logger(__name__)

class GireduceETI(PyrafETI):
    """This class coordinates the external task interface as it relates
    directly to the IRAF task: gireduce
    """
    clparam_dict = None
    ad = None
    def __init__(self, rc, ad):
        """
        Adds the file and parameter objects to a list

        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("GireduceETI __init__")
        PyrafETI.__init__(self, rc)
        self.clparam_dict = {}

        # if ad then it will only process the ad
        self.add_file(InAtList(rc, ad))
        self.add_file(OutAtList(rc, ad))
        self.add_file(LogFile(rc))
        self.add_param(Nbiascontam(rc, ad))
        for param in subtract_overscan_hardcoded_params:
            self.add_param(GireduceParam(rc, param, \
                           subtract_overscan_hardcoded_params[param]))

    def execute(self):
        """Execute pyraf task: gireduce"""
        log.debug("GireduceETI.execute()")

        # Populate object lists
        xcldict = copy(self.clparam_dict)
        for fil in self.file_objs:
            xcldict.update(fil.get_parameter())
        for par in self.param_objs:
            xcldict.update(par.get_parameter())
        iraf.unlearn(iraf.gmos.gireduce)

        # Use setParam to list the parameters in the logfile 
        for par in xcldict:
            #Stderr and Stdout are not recognized by setParam
            if par != "Stderr" and par !="Stdout":
                gemini.gmos.gireduce.setParam(par,xcldict[par])
        log.fullinfo("\nGIREDUCE PARAMETERS:\n")
        iraf.lpar(iraf.gmos.gireduce, Stderr=xcldict["Stderr"], \
            Stdout=xcldict["Stdout"])

        # Execute the task using the same dict as setParam
        # (but this time with Stderr and Stdout) 
        #from pprint import pprint
        #pprint(xcldict)
        try:
            gemini.gmos.gireduce(**xcldict)
        except:
            # catch hard crash
            raise Errors.OutputError("The IRAF task gmos.gireduce failed")
        if gemini.gmos.gireduce.status:
            # catch graceful exit upon error
            raise Errors.OutputError("The IRAF task gmos.gireduce failed")
        else:
            log.fullinfo("The IRAF task gmos.gireduce completed successfully")

    def run(self):
        """Convenience function that runs all the needed operations."""
        log.debug("GireduceETI.run()")
        adlist = []
        self.prepare()
        self.execute()
        adlist = self.recover()
        self.clean()
        return adlist

    def recover(self):
        """Recovers reduction information into memory"""
        log.debug("GireduceETI.recover()")
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
        
        
