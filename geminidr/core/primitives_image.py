from gempy.gemini import gemini_tools as gt

from geminidr.core.primitives_register import Register
from geminidr.core.primitives_resample import Resample
from geminidr.core.parameters_image import ParametersImage

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

    def fringeCorrect(self, adinputs=None, stream='main', **params):
        self.getProcessedFringe(adinputs)
        self.subtractFringe(adinputs)
        return adinputs

    def makeFringe(self, adinputs=None, stream='main', **params):
        return adinputs

    def makeFringeFrame(self, adinputs=None, stream='main', **params):
        return adinputs

    def scaleByIntensity(self, adinputs=None, stream='main', **params):
        return adinputs

    def scaleFringeToScience(self, adinputs=None, stream='main', **params):
        return adinputs

    def subtractFringe(self, adinputs=None, stream='main', **params):
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
        pars = getattr(self.parameters, self.myself())

        fringe_list = pars["fringe"] if pars["fringe"] else [
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
                        aux_type="cal", keyword_comments=self.keyword_comments)
                gt.check_inputs_match(ad, fringe)

            ad.subtract(fringe)

            # Update the header and filename
            ad.phu.set("FLATIM", fringe.filename, self.keyword_comments["FRINGEIM"])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=pars["suffix"],
                                              strip=True)
        return adinputs
