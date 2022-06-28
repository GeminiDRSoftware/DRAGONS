#
#                                                                       DRAGONS
#
#                                                   primitives_igrins_echelle.py
# ------------------------------------------------------------------------------

from geminidr.gemini.lookups import DQ_definitions as DQ
from gempy.gemini import gemini_tools as gt

from geminidr.core.primitives_spect import Spect 
from .primitives_igrins import Igrins 
from . import parameters_igrins_echelle

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------

@parameter_override
class IgrinsEchelle(Igrins, Spect):
    """
    This class contains primitives that applies to IGRINS echelle data.
    """

    tagset = {'GEMINI', 'IGRINS', 'ECHELLE'}

    def __init__(self, adinputs, **kwargs):
        super(IgrinsEchelle, self).__init__(adinputs, **kwargs)
        self.inst_lookups = 'igrinsdr.igrins.lookups'
        self._param_update(parameters_igrins_echelle)

    def myNewPrimitive(self, adinputs=None, **params):
        """
        Description...

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        param2: blah
            blah, blah

        Returns
        -------
        """

        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        # Get params out
        param2 = params['param2']

        # Initialize the list of output AstroData objects
        # It is also possible to modify adinputs in place.
        adoutputs = []

        for ad in adinputs:

            # Do whatever checks on the input are necessary, for example:
            # Check whether this primitive as been run already.
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has"
                            "already been processed by myNewPrimitive".
                            format(ad.filename))
                continue

            # -----------------------
            # DR algorithm goes here
            # -----------------------

            # Timestamp
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)

            adoutputs.append(ad_out)

        return adoutputs

