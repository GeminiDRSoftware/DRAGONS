#
#
#
#                                                  primitives_crossdispersed.py
# -----------------------------------------------------------------------------

from recipe_system.utils.decorators import (parameter_override,
                                            capture_provenance)

from gempy.gemini import gemini_tools as gt
from geminidr.core import Spect, Preprocess
from . import parameters_crossdispersed


@parameter_override
@capture_provenance
class Crossdispersed(Spect, Preprocess):
    """This is the class containing primitives specifically for crossdispersed
    data. It inherits all the primitives from the level above.

    """

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_crossdispersed)

    def flatCorrect(self, adinputs=None, **params):
        """
        This primitive will divide each SCI extension of the inputs by those
        of the corresponding flat. If the inputs contain VAR or DQ frames,
        those will also be updated accordingly due to the division on the data.
        If no flatfield is provided, the calibration database(s) will be
        queried.

        If the flatfield has had a QE correction applied, this information is
        copied into the science header to avoid the correction being applied
        twice.

        This primitive calls the version of flatCorrect in primitives_preprocess
        after first cutting the data into multiple extensions to match the flat.
        Arguments will be passed up to flatCorrect via a call to super().

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        flat: str
            name of flatfield to use
        do_flat: bool
            perform flatfield correction?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        flat = params['flat']

        if flat is None:
            flat_list = self.caldb.get_processed_flat(adinputs)
        else:
            flat_list = (flat, None)

        cut_ads = []
        for ad, flat, origin in zip(*gt.make_lists(adinputs, *flat_list,
                                    force_ad=(1,))):
            if flat is None:
                if 'sq' in self.mode or do_cal == 'force':
                   raise OSError("No processed flat listed for "
                                 f"{ad.filename}")
                else:
                   log.warning(f"No changes will be made to {ad.filename}, "
                               "since no flatfield has been specified")
                   continue

            # Cut the science frames to match the flats, which have already
            # been cut in the flat reduction.
            cut_ads.append(gt.cut_to_match_auxiliary_data(adinput=ad, aux=flat))

        adinputs = super().flatCorrect(adinputs=cut_ads, **params)

        return adinputs
