from copy import copy

from pyraf import iraf
from iraf import gemini
from iraf import gmos
from iraf import gemtools

from gempy.utils import logutils
from gempy.eti_core.pyrafeti import PyrafETI

from .gmosaicfile import InAtList, OutAtList, LogFile
from .gmosaicparam import FlPaste, FlFixpix, Geointer, FlVardq, FlClean, \
    mosaic_detectors_hardcoded_params, GmosaicParam

log = logutils.get_logger(__name__)

class GmosaicETI(PyrafETI):
    """This class coordinates the external task interface as it relates
    directly to the IRAF task: gmosaic
    """
    clparam_dict = None
    ad = None
    def __init__(self, inputs, params, ad):
        """
        Adds the file and parameter objects to a list

        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("GmosaicETI __init__")
        PyrafETI.__init__(self, inputs, params)
        self.clparam_dict = {}

        # if ad then it will only process the ad
        self.add_file(InAtList(inputs, params, ad))
        self.add_file(OutAtList(inputs, params, ad))
        self.add_file(LogFile(inputs, params))
        self.add_param(FlPaste(inputs, params))
        self.add_param(FlFixpix(inputs, params))
        self.add_param(Geointer(inputs, params))
        self.add_param(FlVardq(inputs, params, ad))
        self.add_param(FlClean(inputs, params, ad))
        for param in mosaic_detectors_hardcoded_params:
            self.add_param(GmosaicParam(inputs, params, param, \
                           mosaic_detectors_hardcoded_params[param]))

    def execute(self):
        """Execute pyraf task: gmosaic"""
        log.debug("GmosaicETI.execute()")

        # Populate object lists
        xcldict = copy(self.clparam_dict)
        for fil in self.file_objs:
            xcldict.update(fil.get_parameter())
        for par in self.param_objs:
            xcldict.update(par.get_parameter())
        iraf.unlearn(iraf.gmos.gmosaic)

        # Use setParam to list the parameters in the logfile
        for par in xcldict:
            #Stderr and Stdout are not recognized by setParam
            if par != "Stderr" and par != "Stdout":
                gemini.gmos.gmosaic.setParam(par, xcldict[par])
        log.fullinfo("\nGMOSAIC PARAMETERS:\n")
        iraf.lpar(iraf.gmos.gmosaic, Stderr=xcldict["Stderr"], \
            Stdout=xcldict["Stdout"])

        # Execute the task using the same dict as setParam
        # (but this time with Stderr and Stdout)
        #from pprint import pprint
        #pprint(xcldict)
        try:
            gemini.gmos.gmosaic(**xcldict)
        except:
            # catch hard crash
            raise RuntimeError("The IRAF task gmos.gmosaic failed")
        if gemini.gmos.gmosaic.status:
            # catch graceful exit upon error
            raise RuntimeError("The IRAF task gmos.gmosaic failed")
        else:
            log.fullinfo("The IRAF task gmos.gmosaic completed successfully")

    def run(self):
        """Convenience function that runs all the needed operations."""
        log.debug("GmosaicETI.run()")
        adlist = []
        self.prepare()
        self.execute()
        adlist = self.recover()
        self.clean()
        return adlist

    def recover(self):
        """Recovers reduction information into memory"""
        log.debug("GmosaicETI.recover()")
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
