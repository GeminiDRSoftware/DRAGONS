from __future__ import division
from __future__ import print_function
#
#                                                                  gemini_python
#
#                                                             primitives_gmos.py
# ------------------------------------------------------------------------------
from builtins import zip

import os

import astrodata
import gemini_instruments

# from gempy.gemini.eti import gmosaiceti
from gempy.gemini import gemini_tools as gt
# from gempy.scripts.gmoss_fix_headers import correct_headers
# from gempy.gemini import hdr_fixing as hdrfix

from geminidr.core import CCD
from ..gemini.primitives_gemini import Gemini
from . import parameters_gmos
from .lookups import maskdb

from gemini_instruments.gmos.pixel_functions import get_bias_level

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class GMOS(Gemini, CCD):
    """
    This is the class containing all of the preprocessing primitives
    for the GMOS level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = set(["GEMINI", "GMOS"])

    def __init__(self, adinputs, **kwargs):
        super(GMOS, self).__init__(adinputs, **kwargs)
        self.inst_lookups = 'geminidr.gmos.lookups'
        self._param_update(parameters_gmos)

    def standardizeInstrumentHeaders(self, adinputs=None, suffix=None):
        """
        This primitive is used to make the changes and additions to the
        keywords in the headers of GMOS data, specifically.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by "
                            "standardizeInstrumentHeaders".format(ad.filename))
                adoutputs.append(ad)
                continue

            # Standardize the headers of the input AstroData object. Update the
            # keywords in the headers that are specific to GMOS.
            log.status("Updating keywords that are specific to GMOS")

            # #M Some of the header keywords are wrong for certain types of
            # #M Hamamatsu data. This is temporary fix until GMOS-S DC is fixed
            # if ad.detector_name(pretty=True) == "Hamamatsu-S":
            #     log.status("Fixing headers for GMOS-S Hamamatsu data")
            #     # Image extension headers appear to be correct - MS 2014-10-01
            #     #     correct_image_extensions=Flase
            #     # As does the DATE-OBS but as this seemed to break even after
            #     # apparently being fixed, still perform this check. - MS
            #     hdulist = ad.to_hdulist()
            #     # correct_headers(hdulist, logger=log,
            #     #                 correct_image_extensions=False)
            #     # When we create the new AD object, it needs to retain the
            #     # filename information
            #     orig_path = ad.path
            #     ad = astrodata.open(hdulist)
            #     ad.path = orig_path

            # KL Commissioning GMOS-N Hamamatsu.  Headers are not fully
            # KL settled yet.
            if ad.detector_name(pretty=True) == "Hamamatsu-N":
                log.status("Fixing headers for GMOS-N Hamamatsu data")
                try:
                    ad.phu['DATE-OBS'] = ad.phu['DATE']
                except KeyError:
                    pass

            # Update keywords in the image extensions. The descriptors return
            # the true values on unprepared data.
            descriptors = ['pixel_scale', 'read_noise', 'gain_setting',
                               'gain', 'saturation_level']
            for desc in descriptors:
                keyword = ad._keyword_for(desc)
                comment = self.keyword_comments[keyword]
                dv = getattr(ad, desc)()
                if isinstance(dv, list):
                    for ext, value in zip(ad, dv):
                        ext.hdr.set(keyword, value, comment)
                else:
                    ad.hdr.set(keyword, dv, comment)

            if 'SPECT' in ad.tags:
                kw = ad._keyword_for('dispersion_axis')
                ad.hdr.set(kw, 1, self.keyword_comments[kw])

            # And the bias level too!
            bias_level = get_bias_level(adinput=ad,
                                        estimate='qa' in self.mode)
            for ext, bias in zip(ad, bias_level):
                if bias is not None:
                    ext.hdr.set('RAWBIAS', bias,
                                self.keyword_comments['RAWBIAS'])

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

    def subtractOverscan(self, adinputs=None, **params):
        """
        This primitive subtracts the overscan level from the image. The
        level for each row (currently the primitive requires that the overscan
        region be a vertical strip) is determined in one of the following
        ways, according to the *function* and *order* parameters:

        "poly":   a polynomial of degree *order* (1=linear, etc)
        "spline": using *order* equally-sized cubic spline pieces or, if
                  order=None or 0, a spline that provides a reduced chi^2=1
        "none":   no function is fit, and the value for each row is determined
                  by the overscan pixels in that row

        The fitting is done iteratively but, in the first instance, a running
        median of the rows is calculated and rows that deviate from this median
        are rejected (and used in place of the actual value if function="none")

        The GMOS-specific version of this primitive sets the "nbiascontam" and
        "order" parameters to their Gemini-IRAF defaults if they are None. It
        also removes the bottom 48 (ubinned) rows of the Hamamatsu CCDs from
        consideration in a polynomial fit. It then calls the generic version
        of the primitive.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        niterate: int
            number of rejection iterations
        high_reject: float
            number of standard deviations above which to reject high pixels
        low_reject: float
            number of standard deviations above which to reject low pixels
        overscan_section: str/None
            comma-separated list of IRAF-style overscan sections
        nbiascontam: int/None
            number of columns adjacent to the illuminated region to reject
        function: str
            function to fit ("polynomial" | "spline" | "none")
        order: int
            order of Chebyshev fit or spline/None
        """
        # To avoid crashing at the first line
        if not adinputs:
            return adinputs

        detname = adinputs[0].detector_name(pretty=True)
        func = (params["function"] or 'none').lower()
        if params["order"] is None and func.startswith('poly'):
            params["order"] = 6 if detname.startswith('Hamamatsu') else 0
        if params["nbiascontam"] is None:
            params["nbiascontam"] = 5 if detname == 'e2vDD' else 4

        # Set the overscan_section to ignore the bottom 48 (unbinned) rows
        # if a polynomial fit is being used
        if detname.startswith('Hamamatsu') and func.startswith('poly'):
            for ad in adinputs:
                y1 = 48 // ad.detector_y_bin()
                dsec_list = ad.data_section()
                osec_list = ad.overscan_section()
                for ext, dsec, osec in zip(ad, dsec_list, osec_list):
                    ext.hdr['BIASSEC'] = '[{}:{},{}:{}]'.format(osec.x1+1,
                                                    osec.x2, y1+1, osec.y2)
                    #ext.hdr['DATASEC'] = '[{}:{},{}:{}]'.format(dsec.x1+1,
                    #                                dsec.x2, y1+1, dsec.y2)

        adinputs = super(GMOS, self).subtractOverscan(adinputs, **params)
        return adinputs

    @staticmethod
    def _has_valid_extensions(ad):
        """Check the AD has a valid number of extensions"""
        return len(ad) in [1, 2, 3, 4, 6, 12]

    def _get_bpm_filename(self, ad):
        """
        Gets bad pixel mask for input GMOS science frame.

        Returns
        -------
        str/None: Filename of the appropriate bpms
        """
        log = self.log
        bpm_dir = os.path.join(os.path.dirname(maskdb.__file__), 'BPM')

        inst = ad.instrument()  # Could be GMOS-N or GMOS-S
        xbin = ad.detector_x_bin()
        ybin = ad.detector_y_bin()
        det = ad.detector_name(pretty=True)[:3]
        amps = '{}amp'.format(3 * ad.phu['NAMPS'])
        mos = '_mosaic' if (ad.phu.get(self.timestamp_keys['mosaicDetectors'])
            or ad.phu.get(self.timestamp_keys['tileArrays'])) else ''
        mode_key = '{}_{}_{}{}_{}'.format(inst, det, xbin, ybin, amps)

        db_matches = sorted((k, v) for k, v in maskdb.bpm_dict.items() \
                            if k.startswith(mode_key) and k.endswith(mos))

        # If BPM(s) matched, use the one with the latest version number suffix:
        if db_matches:
            bpm = db_matches[-1][1]
        else:
            log.warning('No BPM found for {}'.format(ad.filename))
            return None

        # Prepend standard path if the filename doesn't start with '/'
        return bpm if bpm.startswith(os.path.sep) else os.path.join(bpm_dir, bpm)
