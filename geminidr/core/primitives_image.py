#
#                                                                  gemini_python
#
#                                                            primitives_image.py
# ------------------------------------------------------------------------------
from gempy.gemini import gemini_tools as gt

from .primitives_register import Register
from .primitives_resample import Resample
from .parameters_image import ParametersImage

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class Image(Register, Resample):
    """
    This is the class containing the generic imaging primitives.
    """
    tagset = set(["IMAGE"])

    def __init__(self, adinputs, **kwargs):
        super(Image, self).__init__(adinputs, **kwargs)
        self.parameters = ParametersImage

    def fringeCorrect(self, adinputs=None, **params):
        self.getProcessedFringe(adinputs)
        adinputs = self.subtractFringe(adinputs, **params)
        return adinputs

    def makeFringe(self, adinputs=None, **params):
        return adinputs

    def makeFringeFrame(self, adinputs=None, **params):
        return adinputs

    def scaleByIntensity(self, adinputs=None, **params):
        return adinputs

    def scaleFringeToScience(self, adinputs=None, **params):
        return adinputs

    def subtractFringe(self, adinputs=None, **params):
        """
        This primitive subtracts a specified fringe frame from the science frame(s)

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        fringe: str/AD
            fringe frame to subtract
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        fringe_list = params["fringe"] if params["fringe"] else [
            self._get_cal(ad, 'processed_fringe') for ad in adinputs]

        # Get a fringe AD object for every science frame
        for ad, fringe in zip(*gt.make_lists(adinputs, fringe_list, force_ad=True)):
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by subtractFringe".
                            format(ad.filename))
                continue

            # Check the inputs have matching filters, binning, and shapes
            try:
                gt.check_inputs_match(ad, fringe)
            except ValueError:
                fringe = gt.clip_auxiliary_data(adinput=ad, aux=fringe,
                                                aux_type="cal")
                gt.check_inputs_match(ad, fringe)

            ad.subtract(fringe)

            # Update the header and filename
            ad.phu.set("FLATIM", fringe.filename, self.keyword_comments["FRINGEIM"])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=params["suffix"],
                                              strip=True)
        return adinputs