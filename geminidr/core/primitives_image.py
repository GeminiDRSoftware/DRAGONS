from gempy.gemini import gemini_tools as gt

from geminidr import PrimitivesBASE
from .parameters_image import ParametersImage

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class Image(PrimitivesBASE):
    """
    This is the class containing the generic imaging primitives.
    (They're not actually very generic)
    """
    tagset = set(["GEMINI", "IMAGE"])

    def __init__(self, adinputs, context, ucals=None, uparms=None):
        super(Image, self).__init__(adinputs, context, ucals=ucals,
                                         uparms=uparms)
        self.parameters = ParametersImage

    def fringeCorrect(self, adinputs=None, stream='main', **params):
        """
        """
        pass

    def makeFringe(self, adinputs=None, stream='main', **params):
        """
        Parameters
        ----------
        subtract_median_image: str
        """
        pass

    def makeFringeFrame(self, adinputs=None, stream='main', **params):
        """
        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        operation: str
        reject_method: str
        subtract_median_image: bool
        """
        pass

    def scaleByIntensity(self, adinputs=None, stream='main', **params):
        """
        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        pass

    def scaleFringeToScience(self, adinputs=None, stream='main', **params):
        """
        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        science: str
        stats_scale: bool
        """
        pass

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
        fringe_param = self.parameters.subtractFringe["fringe"]
        sfx = self.parameters.subtractFringe["suffix"]

        # Get a fringe AD object for every science frame
        for ad, fringe in zip(*gt.make_lists(self.adinputs, fringe_param,
                                             force_ad=True)):
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
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx,
                                              strip=True)
        return
