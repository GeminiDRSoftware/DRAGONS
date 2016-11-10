import astrodata
import gemini_instruments
from gempy.gemini import gemini_tools as gt

from geminidr import PrimitivesBASE
from .parameters_ccd import ParametersCCD

from recipe_system.utils.decorators import parameter_override

import numpy as np
# ------------------------------------------------------------------------------
@parameter_override
class CCD(PrimitivesBASE):
    """
    This is the class containing all of the primitives used to standardize an
    AstroData object.

    """
    tagset = set(["GEMINI"])

    def __init__(self, adinputs, context, ucals=None, uparms=None):
        super(CCD, self).__init__(adinputs, context, ucals=ucals, uparms=uparms)
        self.parameters = ParametersCCD


    def biasCorrect(self, adinputs=None, stream='main', **params):
        self.getProcessedBias()
        self.subtractBias()

    def overscanCorrect(self, adinputs=None, stream='main', **params):
        self.subtractOverscan()
        self.trimOverscan()

    def subtractBias(self, adinputs=None, stream='main', **params):
        """
        The subtractBias primitive will subtract the science extension of the
        input bias frames from the science extension of the input science
        frames. The variance and data quality extension will be updated, if
        they exist.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        bias: str/None
            bias to subtract (None => use default)
        """
        log = self.log
        log.debug(gt.log_message("primitive", "subtractBias", "starting"))
        timestamp_key = self.timestamp_keys["subtractBias"]
        sfx = self.parameters.subtractBias["suffix"]

        #TODO? Assume we're getting filenames, rather than AD instances
        for ad, bias_file in zip(gt.make_lists(adinputs,
                                    self.parameters.subtractBias["bias"])):
            if bias_file is None:
                if 'qa' in self.context:
                    log.warning("No changes will be made to {}, since no "
                                "appropriate bias could be retrieved".
                                format(ad.filename))
                    continue
                else:
                    raise IOError('No processed bias found for {}'.
                                  format(ad.filename))

            bias = astrodata.open(bias_file)
            try:
                gt.check_inputs_match(ad, bias, check_filter=False)
            except ValueError:
                bias = gt.clip_auxiliary_data(ad, bias, aux_type='cal',
                                    keyword_comments=self.keyword_comments)
                gt.check_inputs_match(ad, bias, check_filter=False)

            log.fullinfo('Subtracting this bias from {}:\n{}'.
                         format(ad.filename, bias.filename))
            ad.subtract(bias)
            ad.phu.set('BIASIM', bias.filename, self.keyword_comments['BIASIM'])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
        return

    def subtractOverscan(self, adinputs=None, stream='main', **params):
        pass


    def trimOverscan(self, adinputs=None, stream='main', **params):
        pass

