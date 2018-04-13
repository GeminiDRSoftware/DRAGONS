#
#                                                                  gemini_python
#
#                                                            primitives_image.py
# ------------------------------------------------------------------------------
import numpy as np

from gempy.gemini import gemini_tools as gt
from gempy.library import astrotools as at

from .primitives_register import Register
from .primitives_resample import Resample
from . import parameters_image

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
        self._param_update(parameters_image)

    def fringeCorrect(self, adinputs=None, **params):
        self.getProcessedFringe(adinputs)
        adinputs = self.subtractFringe(adinputs,
                                       **self._inherit_params(params, "subtractFringe",
                                                              pass_suffix=False))
        return adinputs

    def makeFringe(self, adinputs=None, **params):
        return adinputs

    def makeFringeFrame(self, adinputs=None, **params):
        return adinputs

    def scaleByIntensity(self, adinputs=None, **params):
        """
        This primitive scales the inputs so they have the same intensity as
        the reference input (first in the list), which is untouched. Scaling
        can be done by mean or median and a statistics section can be used.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        scaling: str ["mean"/"median"]
            type of scaling to use
        section: str/None
            section of image to use for statistics "x1:x2,y1:y2"
        separate_ext: bool
            if True, scale extensions independently?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        scaling = params["scaling"]
        section = params["section"]
        separate_ext = params["separate_ext"]

        if len(adinputs) < 2:
            log.stdinfo("Scaling has no effect when there are fewer than two inputs")
            return adinputs

        # Do some housekeeping to handle mutually exclusive parameter inputs
        if separate_ext and len(set([len(ad) for ad in adinputs])) > 1:
            log.warning("Scaling by extension requested but inputs have "
                        "different sizes. Turning off.")
            separate_ext = False

        section = at.section_str_to_tuple(section)

        # I'm not making the assumption that all extensions are the same shape
        # This makes things more complicated, but more general
        targets = [np.nan] * len(adinputs[0])
        for ad in adinputs:
            all_data = []
            for index, ext in enumerate(ad):
                extver = ext.hdr['EXTVER']
                if section is None:
                    x1, y1 = 0, 0
                    y2, x2 = ext.data.shape
                else:
                    x1, x2, y1, y2 = section
                data = ext.data[y1:y2, x1:x2]
                if data.size:
                    mask = None if ext.mask is None else ext.mask[y1:y2, x1:x2]
                else:
                    log.warning("Section does not intersect with data for {}:{}."
                                " Using full frame.".format(ad.filename, extver))
                    data = ext.data
                    mask = ext.mask
                if mask is not None:
                    data = data[mask == 0]

                if not separate_ext:
                    all_data.extend(data.ravel())

                if separate_ext or index == len(ad)-1:
                    if separate_ext:
                        value = getattr(np, scaling)(data)
                        log.fullinfo("{}:{} has {} value of {}".format(ad.filename,
                                                            extver, scaling, value))
                    else:
                        value = getattr(np, scaling)(all_data)
                        log.fullinfo("{} has {} value of {}".format(ad.filename,
                                                                    scaling, value))
                    if np.isnan(targets[index]):
                        targets[index] = value
                    else:
                        factor = targets[index] / value
                        log.fullinfo("Multiplying by {}".format(factor))
                        if separate_ext:
                            ext *= factor
                        else:
                            ad *= factor

            ad.update_filename(suffix=params["suffix"], strip=True)
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
            ad.update_filename(suffix=params["suffix"], strip=True)
        return adinputs