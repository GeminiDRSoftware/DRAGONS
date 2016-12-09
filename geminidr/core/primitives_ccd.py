import astrodata
import gemini_instruments
from gempy.gemini import gemini_tools as gt
from gempy.gemini.eti import gireduceeti

from .. import PrimitivesBASE
from .parameters_ccd import ParametersCCD

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class CCD(PrimitivesBASE):
    """
    This is the class containing all of the primitives used for generic CCD
    reduction.
    """
    tagset = None

    def __init__(self, adinputs, context, ucals=None, uparms=None):
        super(CCD, self).__init__(adinputs, context, ucals=ucals, uparms=uparms)
        self.parameters = ParametersCCD

    def biasCorrect(self, adinputs=None, stream='main', **params):
        self.getProcessedBias(adinputs)
        adinputs = self.subtractBias(adinputs)
        return adinputs

    def overscanCorrect(self, adinputs=None, stream='main', **params):
        adinputs = self.subtractOverscan(adinputs)
        adinputs = self.trimOverscan(adinputs)
        return adinputs

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
        bias: str/list of str
            bias(es) to subtract
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        pars = getattr(self.parameters, self.myself())

        bias_list = pars["bias"] if pars["bias"] else [
            self._get_cal(ad, 'processed_bias') for ad in adinputs]

        # Provide a bias AD object for every science frame
        for ad, bias in zip(*gt.make_lists(adinputs, bias_list, force_ad=True)):
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by subtractBias".
                            format(ad.filename))
                continue

            if bias is None:
                if 'qa' in self.context:
                    log.warning("No changes will be made to {}, since no "
                                "bias was specified".format(ad.filename))
                    continue
                else:
                    raise IOError('No processed bias listed for {}'.
                                  format(ad.filename))

            try:
                gt.check_inputs_match(ad, bias, check_filter=False)
            except ValueError:
                bias = gt.clip_auxiliary_data(ad, bias, aux_type='cal',
                                    keyword_comments=self.keyword_comments)
                # An Error will be raised if they don't match now
                gt.check_inputs_match(ad, bias, check_filter=False)

            log.fullinfo('Subtracting this bias from {}:\n{}'.
                         format(ad.filename, bias.filename))
            ad.subtract(bias)

            # Record bias used, timestamp, and update filename
            ad.phu.set('BIASIM', bias.filename, self.keyword_comments['BIASIM'])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=pars["suffix"],
                                              strip=True)
        return adinputs

    def subtractOverscan(self, adinputs=None, stream='main', **params):
        """
        This primitive uses External Task Interface to gireduce to subtract
        the overscan from the input images.

        Variance and DQ planes, if they exist, will be saved and restored
        after gireduce has been run.

        NOTE:
        The inputs to this function MUST be prepared.

        Parameters
        ----------
        overscan_section: str/None
            comma-separated list of IRAF-style overscan sections
            None => use nbiascontam=4 columns
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        # Need to create a new output list since the ETI makes new AD objects
        adoutputs = []
        for ad in adinputs:
            if (ad.phu.get('GPREPARE') is None and
                        ad.phu.get('PREPARE') is None):
                raise IOError('{} must be prepared'.format(ad.filename))
            if ad.phu.get(timestamp_key) is not None:
                log.warning('No changes will be made to {}, since it has '
                            'already been processed by subtractOverscan'.
                            format(ad.filename))
                adoutputs.append(ad)
                continue

            gireduce_task = gireduceeti.GireduceETI([],
                                        self.parameters.subtractOverscan, ad)
            adout = gireduce_task.run()
            # Need to reattach DQ, VAR, and other bits'n'bobs
            for extout, extin in zip(adout, ad):
                extout.reset(extout.data, extin.mask, extin.variance)
                if hasattr(extin, 'OBJCAT'):
                    extout.OBJCAT = extin.OBJCAT
                if hasattr(extin, 'OBJMASK'):
                    extout.OBJMASK = extin.OBJMASK
            if hasattr(ad, 'REFCAT'):
                adout.REFCAT = ad.REFCAT
            gt.mark_history(adout, primname=self.myself(), keyword=timestamp_key)
            adoutputs.append(adout)

        # Reset inputs to the ETI outputs
        adinputs = adoutputs
        return adinputs

    def trimOverscan(self, adinputs=None, stream='main', **params):
        """
        The trimOverscan primitive trims the overscan region from the input
        AstroData object and updates the headers.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = self.parameters.trimOverscan["suffix"]

        for ad in adinputs:
            if ad.phu.get(timestamp_key) is not None:
                log.warning('No changes will be made to {}, since it has '
                            'already been processed by trimOverscan'.
                            format(ad.filename))
                continue

            ad = gt.trim_to_data_section(ad,
                                    keyword_comments=self.keyword_comments)

            # Set keyword, timestamp, and update filename
            ad.phu.set('TRIMMED', 'yes', self.keyword_comments['TRIMMED'])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
        return adinputs
