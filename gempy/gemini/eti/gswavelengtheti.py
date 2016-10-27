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
from gswavelengthfile import InAtList, OutDatabase, LogFile
from gswavelengthparam import hardcoded_params, FlInter, GswavelengthParam
    
log = logutils.get_logger(__name__)

class GswavelengthETI(PyrafETI):
    """This class coordinates the external task interface as it relates
    directly to the IRAF task: gswavelength
    """
    clparam_dict = None
    ad = None
    def __init__(self, rc, ad):
        """
        Adds the file and parameter objects to a list

        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("GswavelengthETI __init__")
        PyrafETI.__init__(self, rc)
        self.clparam_dict = {}

        # if ad then it will only process the ad
        self.add_file(InAtList(rc, ad))
        self.add_file(OutDatabase(rc, ad))
        self.add_file(LogFile(rc))
        self.add_param(FlInter(rc, ad))
        for param in hardcoded_params:
            self.add_param(GswavelengthParam(rc, param,
                                             hardcoded_params[param]))

    def execute(self):
        """Execute pyraf task: gswavelength"""
        log.debug("GswavelengthETI.execute()")

        # Populate object lists
        xcldict = copy(self.clparam_dict)
        for fil in self.file_objs:
            xcldict.update(fil.get_parameter())
        for par in self.param_objs:
            xcldict.update(par.get_parameter())
        iraf.unlearn(iraf.gmos.gswavelength)

        # Use setParam to list the parameters in the logfile 
        for par in xcldict:
            #Stderr and Stdout are not recognized by setParam
            if par != "Stderr" and par !="Stdout":
                gemini.gmos.gswavelength.setParam(par,xcldict[par])
        log.fullinfo("\nGSWAVELENGTH PARAMETERS:\n")
        iraf.lpar(iraf.gmos.gswavelength, Stderr=xcldict["Stderr"], \
            Stdout=xcldict["Stdout"])

        # Execute the task using the same dict as setParam
        # (but this time with Stderr and Stdout) 
        #from pprint import pprint
        #pprint(xcldict)
        try:
            gemini.gmos.gswavelength(**xcldict)
        except iraf.IrafError:
            gemini.gmos.gswavelength.status = 1
        if gemini.gmos.gswavelength.status:
            raise Errors.OutputError("The IRAF task gmos.gswavelength failed")
        else:
            log.fullinfo("The IRAF task gmos.gswavelength completed successfully")

    def run(self):
        """Convenience function that runs all the needed operations."""
        log.debug("GswavelengthETI.run()")
        adlist = []
        self.prepare()
        self.execute()
        adlist = self.recover()
        self.clean()
        return adlist

    def recover(self):
        """Recovers reduction information into memory"""
        log.debug("GswavelengthETI.recover()")
        adlist = []
        for par in self.param_objs:
            par.recover()
        for fil in self.file_objs:
            if isinstance(fil, OutDatabase):
                adlist.extend(fil.recover())
            else:
                fil.recover()
        if len(adlist) == 1:
            return adlist[0]
        else:
            return adlist
        
        
