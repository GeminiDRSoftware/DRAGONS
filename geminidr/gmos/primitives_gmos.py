from __future__ import division
from __future__ import print_function
#
#                                                                  gemini_python
#
#                                                             primitives_gmos.py
# ------------------------------------------------------------------------------
from builtins import zip

import os
import numpy as np
from copy import deepcopy

import astrodata
import gemini_instruments

#from gempy.gemini.eti import gmosaiceti
from gempy.gemini import gemini_tools as gt
#from gempy.scripts.gmoss_fix_headers import correct_headers
from gempy.gemini import hdr_fixing as hdrfix

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

    # def mosaicDetectors(self, adinputs=None, **params):
    #     """
    #     This primitive will mosaic the frames of the input images. It uses
    #     the the ETI and pyraf to call gmosaic from the gemini IRAF package.
    #
    #     Parameters
    #     ----------
    #     suffix: str
    #         suffix to be added to output files
    #     tile: bool
    #         tile images instead of a proper mosaic?
    #     interpolate_gaps: bool
    #         interpolate across gaps?
    #     interpolator: str
    #         type of interpolation to use across chip gaps
    #         (linear, nearest, poly3, poly5, spline3, sinc)
    #     """
    #     log = self.log
    #     log.debug(gt.log_message("primitive", self.myself(), "starting"))
    #     timestamp_key = self.timestamp_keys[self.myself()]
    #
    #     adoutputs = []
    #     for ad in adinputs:
    #         # Data validation
    #         if 'PREPARED' not in ad.tags:
    #             raise IOError('{} must be prepared'.format(ad.filename))
    #
    #         if ad.phu.get(timestamp_key):
    #             log.warning("No changes will be made to {}, since it has "
    #                         "already been processed by mosaicDetectors".
    #                         format(ad.filename))
    #             continue
    #
    #         if len(ad) == 1:
    #             log.stdinfo("No changes will be made to {}, since it "
    #                         "contains only one extension".format(ad.filename))
    #             continue
    #
    #         # Save keywords for restoration after gmosaic
    #         bunit = set(ad.hdr.get('BUNIT'))
    #         if len(bunit) > 1:
    #             raise IOError("BUNIT needs to be the same for all extensions")
    #         else:
    #             bunit = bunit.pop()
    #         try:
    #             avg_overscan = np.mean([overscan for overscan in
    #                             ad.hdr.get('OVERSCAN') if overscan is not None])
    #         except TypeError:
    #             avg_overscan = None
    #         all_ampname = ','.join(ampname for ampname in ad.hdr.get('AMPNAME')
    #                                if ampname is not None)
    #
    #         old_detsec = min(ad.detector_section(), key=lambda x: x.x1)
    #
    #         # Instantiate ETI and then run the task
    #         gmosaic_task = eti.gmosaiceti.GmosaicETI([], params, ad)
    #         ad_out = gmosaic_task.run()
    #
    #         # Get new DATASEC keyword, using the full shape
    #         data_shape = ad_out[0].data.shape
    #         new_datasec = "[1:{1},1:{0}]".format(*data_shape)
    #
    #         # Make new DETSEC keyword
    #         xbin = ad_out.detector_x_bin()
    #         unbin_width = data_shape[1] if xbin is None else data_shape[1]*xbin
    #         new_detsec = "" if old_detsec is None else "[{}:{},{}:{}]".format(
    #                                 old_detsec.x1+1, old_detsec.x1+unbin_width,
    #                                 old_detsec.y1+1, old_detsec.y2)
    #
    #         # Truncate long comments to avoid an error
    #         if all_ampname is not None:
    #             ampcomment = self.keyword_comments["AMPNAME"]
    #             if len(all_ampname)>=65:
    #                 ampcomment = ""
    #             else:
    #                 ampcomment = ampcomment[0:65-len(all_ampname)]
    #         else:
    #             ampcomment = ""
    #
    #         # Restore keywords to extension header
    #         if bunit:
    #             ad_out.hdr.set('BUNIT', bunit, self.keyword_comments["BUNIT"])
    #         if avg_overscan:
    #             ad_out.hdr.set('OVERSCAN', avg_overscan,
    #                            comment=self.keyword_comments["OVERSCAN"])
    #         if all_ampname:
    #             ad_out.hdr.set('AMPNAME', all_ampname, comment=ampcomment)
    #         ad_out.hdr.set("DETSEC", new_detsec,
    #                           comment=self.keyword_comments["DETSEC"])
    #         ad_out.hdr.set("CCDSEC", new_detsec,
    #                           comment=self.keyword_comments["CCDSEC"])
    #         ad_out.hdr.set("DATASEC", new_datasec,
    #                           comment=self.keyword_comments["DATASEC"])
    #         ad_out.hdr.set("CCDNAME", ad.detector_name(),
    #                           comment=self.keyword_comments["CCDNAME"])
    #
    #         if hasattr(ad, 'REFCAT'):
    #             ad_out.REFCAT = deepcopy(ad.REFCAT)
    #
    #         gt.mark_history(ad_out, primname=self.myself(), keyword=timestamp_key)
    #         adoutputs.append(ad_out)
    #     return adoutputs

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

        adoutputs = []
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

            ##M Some of the header keywords are wrong for certain types of
            ##M Hamamatsu data. This is temporary fix until GMOS-S DC is fixed
#             if ad.detector_name(pretty=True) == "Hamamatsu-S":
#                 log.status("Fixing headers for GMOS-S Hamamatsu data")
#                 # Image extension headers appear to be correct - MS 2014-10-01
#                 #     correct_image_extensions=Flase
#                 # As does the DATE-OBS but as this seemed to break even after
#                 # apparently being fixed, still perform this check. - MS
#                 hdulist = ad.to_hdulist()
# #                correct_headers(hdulist, logger=log,
# #                                correct_image_extensions=False)
#                 # When we create the new AD object, it needs to retain the
#                 # filename information
#                 orig_path = ad.path
#                 ad = astrodata.open(hdulist)
#                 ad.path = orig_path

            # KL Commissioning GMOS-N Hamamatsu.  Headers are not fully
            # KL settled yet.
            if ad.detector_name(pretty=True) == "Hamamatsu-N":
                log.status("Fixing headers for GMOS-N Hamamatsu data")
                hdulist = ad.to_hdulist()
                updated = hdrfix.gmosn_ham_fixes(hdulist, verbose=False)
                # When we create a new AD object, it needs to retain the
                # filename information
                if updated:
                    orig_path = ad.path
                    ad = astrodata.open(hdulist)
                    ad.path = orig_path

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
            adoutputs.append(ad)
        return adoutputs

    def subtractOverscan(self, adinputs=None, **params):
        """
        Subtract the overscan level from the image by fitting a polynomial
        to the overscan region. This sets the appropriate parameters for GMOS
        (the gireduce defaults) and calls the CCD-level method.

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
        detname = adinputs[0].detector_name(pretty=True)
        func = (params["function"] or 'none').lower()
        if params["order"] is None and func.startswith('poly'):
            params["order"] = 6 if detname.startswith('Hamamatsu') else 0
        if params["nbiascontam"] is None:
            params["nbiascontam"] = 5 if detname == 'e2vDD' else 4

        # Set the overscan_section and data_section keywords to chop off the
        # bottom 48 (unbinned) rows, as Gemini-IRAF does
        if detname.startswith('Hamamatsu') and func.startswith('poly'):
            for ad in adinputs:
                y1 = 48 // ad.detector_y_bin()
                dsec_list = ad.data_section()
                osec_list = ad.overscan_section()
                for ext, dsec, osec in zip(ad, dsec_list, osec_list):
                    ext.hdr['BIASSEC'] = '[{}:{},{}:{}]'.format(osec.x1+1,
                                                    osec.x2, y1+1, osec.y2)
                    ext.hdr['DATASEC'] = '[{}:{},{}:{}]'.format(dsec.x1+1,
                                                    dsec.x2, y1+1, dsec.y2)

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
        key = '{}_{}_{}{}_{}_{}{}'.format(inst, det, xbin, ybin, amps,
                                          'v1', mos)
        try:
            bpm = maskdb.bpm_dict[key]
        except:
            log.warning('No BPM found for {}'.format(ad.filename))
            return None

        # Prepend standard path if the filename doesn't start with '/'
        return bpm if bpm.startswith(os.path.sep) else os.path.join(bpm_dir, bpm)
