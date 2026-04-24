#                                                                  gemini_python
#
#                                                    primitives_igrins_bundle.py
# ------------------------------------------------------------------------------
import astrodata, gemini_instruments
from .primitives_igrins import Igrins
from . import parameters_igrins_bundle
from .lookups.timestamp_keywords import timestamp_keys

from gempy.gemini import gemini_tools as gt
from recipe_system.utils.decorators import parameter_override, capture_provenance


@parameter_override
@capture_provenance
class IgrinsBundle(Igrins):
    """
    Primitives for unpacking IGRINS observation bundle files.
    """
    tagset = set(["GEMINI", "IGRINS", "BUNDLE"])

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_igrins_bundle)
        self.timestamp_keys.update(timestamp_keys)

    def splitBundle(self, adinputs=None, **params):
        """
        Break a raw IGRINS file into separate H and K AD objects.
        Additional extensions (e.g., the slit viewer) are ignored.

        It does not check that there is only (at most) one of each type in
        a bundle (if there is, multiple files will have the same filename).
        It deliberately does not require there to be an H *and* K in each
        bundle, so data can still be processed if one camera is not working.
        """
        log = self.log
        log.debug(gt.log_message('primitive', self.myself(), 'starting'))
        timestamp_key = self.timestamp_keys[self.myself()]

        adoutputs = {'H': [], 'K': []}
        for ad in adinputs:
            kw = ad._keyword_for('wavelength_band')
            log.stdinfo(f"Splitting {ad.filename}")
            for ext, band in zip(ad, ad.hdr.get(kw)):
                if band in adoutputs:
                    adout = astrodata.create(ad.phu)  # deepcopied
                    new_filename = ad.filename.replace('.fits',
                                                       f'_{band}.fits')
                    adout.phu['ORIGNAME'] = new_filename
                    adout.filename = new_filename
                    adout.append(ext)

                    # Add filter to PHU
                    adout.phu[kw] = band

                    gt.mark_history(adout, primname=self.myself(),
                                    keyword=timestamp_key)
                    adoutputs[band].append(adout)

        return adoutputs['H'] + adoutputs['K']
