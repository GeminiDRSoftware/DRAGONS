import sys
from copy import copy

from pyraf import iraf
from iraf import gemini
from iraf import gemtools

from astrodata.utils import Errors
from astrodata.utils import logutils
from astrodata.utils.gemutil import pyrafLoader
from astrodata.eti.pyrafeti import PyrafETI
from gscrrejfile import InFile, OutFile, LogFile
from gscrrejparam import hardcoded_params, GscrrejParam

log = logutils.get_logger(__name__)

class GscrrejETI(PyrafETI):
    """This class coordinates the external task interface as it relates
    directly to the IRAF task: gscrrej
    """
    clparam_dict = None
    def __init__(self, rc=None, ad=None):
        """
        Adds the file and parameter objects to a list

        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("GscrrejETI __init__")
        PyrafETI.__init__(self, rc)
        self.clparam_dict = {}
        
        if ad is not None:
            adinput = [ad]
        else:
            adinput = rc.get_inputs_as_astrodata()
        for ad in adinput:
            self.add_file(InFile(rc,ad))
            self.add_file(OutFile(rc,ad))
        self.add_file(LogFile(rc))
        for param in hardcoded_params:
            self.add_param(GscrrejParam(rc, param, hardcoded_params[param]))

    def execute(self):
        """Execute pyraf task: gscrrej"""
        log.debug("GscrrejETI.execute()")

        # This task can only work on one file at a time, so loop over
        # the infiles
        for i in range(len(self.file_objs)):

            # Get infile
            fil = self.file_objs[i]
            if isinstance(fil, InFile):
                # Get default parameters
                xcldict = copy(self.clparam_dict)
                for par in self.param_objs:
                    xcldict.update(par.get_parameter())
                iraf.unlearn(iraf.gscrrej)

                # Get outfile
                outfil = self.file_objs[i+1]

                # Set in/out file parameters
                xcldict.update(fil.get_parameter())
                xcldict.update(outfil.get_parameter())

                # Use setParam to list the parameters in the logfile 
                for par in xcldict:
                    #Stderr and Stdout are not recognized by setParam
                    if par != "Stderr" and par !="Stdout":
                        gemini.gscrrej.setParam(par,xcldict[par])
                log.fullinfo("\nGSCRREJ PARAMETERS:\n")
                iraf.lpar(iraf.gscrrej, Stderr=xcldict["Stderr"], \
                              Stdout=xcldict["Stdout"])

                # Execute the task using the same dict as setParam
                # (but this time with Stderr and Stdout) 
                gemini.gscrrej(**xcldict)
                if gemini.gscrrej.status:
                    raise Errors.OutputError("The IRAF task gscrrej failed")
                else:
                    log.fullinfo("The IRAF task gscrrej completed successfully")
            else:
                continue

    def run(self):
        """Convenience function that runs all the needed operations."""
        log.debug("GscrrejETI.run()")
        self.prepare()
        self.execute()
        adlist = self.recover()
        self.clean()
        return adlist

    def recover(self):
        """Recovers reduction information into memory"""
        log.debug("GscrrejETI.recover()")
        adlist = []
        for par in self.param_objs:
            par.recover()
        for fil in self.file_objs:
            if isinstance(fil, OutFile):
                adlist.append(fil.recover())
            else:
                fil.recover()
        if len(adlist) == 1:
            return adlist[0]
        else:
            return adlist
