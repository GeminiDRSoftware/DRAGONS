import sys
from copy import copy

from pyraf import iraf

from astrodata.utils import Errors
from astrodata.utils import logutils
from astrodata.utils.gemutil import pyrafLoader
from astrodata.eti.pyrafeti import PyrafETI
from splotfile import InAtList
from splotparam import hardcoded_params, SplotParam

log = logutils.get_logger(__name__)

class SplotETI(PyrafETI):
    """This class coordinates the external task interface as it relates
    directly to the IRAF task: splot
    """
    clparam_dict = None
    def __init__(self, rc=None, ad=None):
        """
        Adds the file and parameter objects to a list

        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("SplotETI __init__")
        PyrafETI.__init__(self, rc)
        self.clparam_dict = {}
        self.add_file(InAtList(rc,ad))
        for param in hardcoded_params:
            self.add_param(SplotParam(rc, param, hardcoded_params[param]))

    def execute(self):
        """Execute pyraf task: splot"""
        log.debug("SplotETI.execute()")

        # Populate object lists
        xcldict = copy(self.clparam_dict)
        for fil in self.file_objs:
            xcldict.update(fil.get_parameter())
        for par in self.param_objs:
            xcldict.update(par.get_parameter())
        iraf.unlearn(iraf.splot)

        # Use setParam to list the parameters in the logfile 
        for par in xcldict:
            #Stderr and Stdout are not recognized by setParam
            if par != "Stderr" and par !="Stdout":
                iraf.splot.setParam(par,xcldict[par])
        log.fullinfo("\nSPLOT PARAMETERS:\n")
        iraf.lpar(iraf.splot, Stderr=xcldict["Stderr"], \
            Stdout=xcldict["Stdout"])

        # Execute the task using the same dict as setParam
        # (but this time with Stderr and Stdout) 
        iraf.splot(**xcldict)

    def run(self):
        """Convenience function that runs all the needed operations."""
        log.debug("SplotETI.run()")
        self.prepare()
        self.execute()
        self.recover()
        self.clean()

    def recover(self):
        """Recovers reduction information into memory"""
        log.debug("SplotETI.recover()")
        adlist = []
        for par in self.param_objs:
            par.recover()
        for fil in self.file_objs:
            fil.recover()
