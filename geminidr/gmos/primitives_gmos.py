#
#                                                                  gemini_python
#
#                                                             primitives_gmos.py
# ------------------------------------------------------------------------------
import os
import numpy as np
from copy import deepcopy

import astrodata
import gemini_instruments

from gempy.gemini import eti
from gempy.gemini import gemini_tools as gt
#from gempy.scripts.gmoss_fix_headers import correct_headers
from gempy.gemini import hdr_fixing as hdrfix

from geminidr.core import CCD
from ..gemini.primitives_gemini import Gemini
from .parameters_gmos import ParametersGMOS
from .lookups.array_gaps import gmosArrayGaps
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
        self.parameters = ParametersGMOS

    def mosaicDetectors(self, adinputs=None, **params):
        """
        This primitive will mosaic the frames of the input images. It uses
        the the ETI and pyraf to call gmosaic from the gemini IRAF package.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        tile: bool
            tile images instead of a proper mosaic?
        interpolate_gaps: bool
            interpolate across gaps?
        interpolator: str
            type of interpolation to use across chip gaps
            (linear, nearest, poly3, poly5, spline3, sinc)
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        adoutputs = []
        for ad in adinputs:
            # Data validation
            if ad.phu.get('GPREPARE') is None and ad.phu.get('PREPARE') is None:
                raise IOError('{} must be prepared'.format(ad.filename))

            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by mosaicDetectors".
                            format(ad.filename))
                continue

            if len(ad) == 1:
                log.stdinfo("No changes will be made to {}, since it "
                            "contains only one extension".format(ad.filename))
                continue

            # Save keywords for restoration after gmosaic
            bunit = set(ad.hdr.get('BUNIT'))
            if len(bunit) > 1:
                raise IOError("BUNIT needs to be the same for all extensions")
            else:
                bunit = bunit.pop()
            try:
                avg_overscan = np.mean([overscan for overscan in
                                ad.hdr.get('OVERSCAN') if overscan is not None])
            except TypeError:
                avg_overscan = None
            all_ampname = ','.join(ampname for ampname in ad.hdr.get('AMPNAME')
                                   if ampname is not None)

            old_detsec = min(ad.detector_section(), key=lambda x: x.x1)

            # Instantiate ETI and then run the task
            gmosaic_task = eti.gmosaiceti.GmosaicETI([], params, ad)
            ad_out = gmosaic_task.run()

            # Get new DATASEC keyword, using the full shape
            data_shape = ad_out[0].data.shape
            new_datasec = "[1:{1},1:{0}]".format(*data_shape)

            # Make new DETSEC keyword
            xbin = ad_out.detector_x_bin()
            unbin_width = data_shape[1] if xbin is None else data_shape[1]*xbin
            new_detsec = "" if old_detsec is None else "[{}:{},{}:{}]".format(
                                    old_detsec.x1+1, old_detsec.x1+unbin_width,
                                    old_detsec.y1+1, old_detsec.y2)

            # Truncate long comments to avoid an error
            if all_ampname is not None:
                ampcomment = self.keyword_comments["AMPNAME"]
                if len(all_ampname)>=65:
                    ampcomment = ""
                else:
                    ampcomment = ampcomment[0:65-len(all_ampname)]
            else:
                ampcomment = ""

            # Restore keywords to extension header
            if bunit:
                ad_out.hdr.set('BUNIT', bunit, self.keyword_comments["BUNIT"])
            if avg_overscan:
                ad_out.hdr.set('OVERSCAN', avg_overscan,
                               comment=self.keyword_comments["OVERSCAN"])
            if all_ampname:
                ad_out.hdr.set('AMPNAME', all_ampname, comment=ampcomment)
            ad_out.hdr.set("DETSEC", new_detsec,
                              comment=self.keyword_comments["DETSEC"])
            ad_out.hdr.set("CCDSEC", new_detsec,
                              comment=self.keyword_comments["CCDSEC"])
            ad_out.hdr.set("DATASEC", new_datasec,
                              comment=self.keyword_comments["DATASEC"])
            ad_out.hdr.set("CCDNAME", ad.detector_name(),
                              comment=self.keyword_comments["CCDNAME"])

            if hasattr(ad, 'REFCAT'):
                ad_out.REFCAT = deepcopy(ad.REFCAT)

            gt.mark_history(ad_out, primname=self.myself(), keyword=timestamp_key)
            adoutputs.append(ad_out)
        return adoutputs

    def standardizeInstrumentHeaders(self, adinputs=None, **params):
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
                                        estimate='qa' in self.context)
            for ext, bias in zip(ad, bias_level):
                if bias is not None:
                    ext.hdr.set('RAWBIAS', bias,
                                self.keyword_comments['RAWBIAS'])

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=params["suffix"],
                                              strip=True)
            adoutputs.append(ad)
        return adoutputs

    def tileArrays(self, adinputs=None, **params):
        """
        This tiles the GMOS detectors together

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        tile_all: bool
            tile to a single extension (as opposed to one extn per CCD)?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        tile_all = params["tile_all"]

        adoutputs = []
        for ad in adinputs:
            # Start building output AD object with the input PHU
            adoutput = astrodata.create(ad.header[0])
#            out_hdulist = ad.to_hdulist()[:1]

            # Do nothing if there is only one science extension
            if len(ad) == 1:
                log.fullinfo("Only one science extension found; "
                             "no tiling done for {}".format(ad.filename))
                adoutputs.append(ad)
                continue

            # First trim off any overscan regions still present
            # so they won't get tiled with science data
            log.fullinfo("Trimming data to data section:")
            old_shape = [ext.data.shape for ext in ad]
            ad = gt.trim_to_data_section(deepcopy(ad),
                                         keyword_comments=self.keyword_comments)
            new_shape = [ext.data.shape for ext in ad]
            changed = old_shape!=new_shape

            # Make chip gaps to tile with science extensions if tiling all
            # Gap width comes from a lookup table
            gap_height = int(ad[0].data.shape[0])
            gap_width = _obtain_arraygap(ad)
            chip_gap = np.zeros((gap_height,gap_width), dtype=ad[0].data.dtype)

            # Get the correct order of the extensions by sorting on
            # the first element in detector section
            # (raw ordering is whichever amps read out first)
            ampsorder = np.argsort([detsec.x1
                                    for detsec in ad.detector_section()])
            in_order = all(ampsorder == np.arange(len(ad)))

            # Get array sections for determining when a new array is found
            ccdx1 = np.array([ccdsec.x1 for ccdsec in ad.array_section()])[ampsorder]

            # Make a list of the output extensions where each array ends up
            num_ccd = 1
            ccd_map = [num_ccd]
            for i in range(1, len(ccdx1)):
                if ccdx1[i]<=ccdx1[i-1]:
                    num_ccd += 1
                ccd_map.append(num_ccd)
            if num_ccd==len(ad) and in_order and not tile_all:
                log.fullinfo("Only one amplifier per array; no tiling done "
                             "for {}".format(ad.filename))
                # If the file has been trimmed, it needs to be timestamped later
                if changed:
                    adoutput = ad
                else:
                    # Otherwise we can move onto the next adinput
                    adoutputs.append(ad)
                    continue
            else:
                if not in_order:
                    log.fullinfo("Reordering data by detector section")
                if tile_all:
                    log.fullinfo("Tiling all data into one extension")
                elif num_ccd != len(ad):
                    log.fullinfo("Tiling data into one extension per array")

                chip_xshift = 0  # Shift due to CCDs on left of reference
                ccd_map = np.array(ccd_map)
                for ccd in range(1, num_ccd+1):
                    amps_on_ccd = ampsorder[ccd_map==ccd]
                    extns = [ad[i] for i in amps_on_ccd]
                    # Use the centre-left amplifier's HDU as basis for new HDU
                    ref_ext = amps_on_ccd[(len(amps_on_ccd) - 1) // 2]
                    # Stack the data, etc.
                    data = np.hstack([ext.data for ext in extns])
                    mask = None if any(ext.mask is None for ext in extns) \
                        else np.hstack([ext.mask for ext in extns])
                    var = None if any(ext.variance is None for ext in extns) \
                        else np.hstack([ext.variance for ext in extns])
                    try:
                        objmask = np.hstack([ext.OBJMASK for ext in extns])
                    except AttributeError:
                        objmask = None

                    # Store this information from the leftmost extension
                    if ccd==1 or not tile_all:
                        old_detsec = extns[0].detector_section()
                        old_ccdsec = extns[0].array_section()

                    # Add the widths of all arrays to the left of the reference
                    xshift = sum(ext.data.shape[1] for
                                 ext in extns[:int((len(amps_on_ccd) - 1) // 2)])

                    if tile_all and ccd>1:
                        # Set reference extension to be the centre-left of all
                        ref_ext = ampsorder[(len(ampsorder) - 1) // 2]
                        # Calculate total horizontal shift if the reference
                        # array is on this CCD
                        if ref_ext in amps_on_ccd:
                            chip_xshift += all_data.shape[1] + chip_gap.shape[1]

                        # Add a gap and this CCD to the existing tiled data
                        all_data = np.hstack([all_data, chip_gap, data])
                        if all_mask is not None and mask is not None:
                            all_mask = np.hstack([all_mask,
                                        chip_gap.astype(np.int16)+16, mask])
                        else:
                            all_mask = None
                        if all_var is not None and var is not None:
                            all_var = np.hstack([all_var, chip_gap, var])
                        else:
                            all_var = None
                        if all_objmask is not None and objmask is not None:
                            all_objmask = np.hstack([all_objmask, chip_gap,
                                                     objmask])
                        else:
                            all_objmask = None
                        ampslist.extend(ad[i].array_name() for i in amps_on_ccd)
                    else:
                        all_data = data
                        all_mask = mask
                        all_var = var
                        all_objmask = objmask
                        ampslist = [ad[i].array_name() for i in amps_on_ccd]

                    if ccd==num_ccd or not tile_all:
                        # Append what we've got. Base it on the reference extn
                        ext_to_add = deepcopy(ad[ref_ext])
                        ext_to_add[0].reset(all_data, all_mask, all_var)
                        if all_objmask is not None:
                            ext_to_add[0].OBJMASK = all_objmask

                        # Update keywords in the header
                        ext_to_add.hdr.set('CCDNAME', ad.detector_name(),
                                           self.keyword_comments['CCDNAME'])
                        ext_to_add.hdr.set('AMPNAME', ','.join(ampslist),
                                           self.keyword_comments['AMPNAME'])

                        data_shape = ext_to_add[0].data.shape
                        new_datasec = '[1:{1},1:{0}]'.format(*data_shape)
                        ext_to_add.hdr.set('DATASEC', new_datasec,
                                           self.keyword_comments['DATASEC'])

                        unbin_width = data_shape[1] * ad.detector_x_bin()
                        new_detsec = '[{}:{},{}:{}]'.format(old_detsec.x1+1,
                                    old_detsec.x1+unbin_width, old_detsec.y1+1,
                                                            old_detsec.y2)
                        ext_to_add.hdr.set('DETSEC', new_detsec,
                                           self.keyword_comments['DETSEC'])

                        new_ccdsec = '[{}:{},{}:{}]'.format(old_ccdsec.x1+1,
                                    old_ccdsec.x1+unbin_width, old_ccdsec.y1+1,
                                                            old_ccdsec.y2)
                        ext_to_add.hdr.set('CCDSEC', new_ccdsec,
                                           self.keyword_comments['CCDSEC'])

                        crpix1 = ext_to_add.hdr.get('CRPIX1')[0]
                        if crpix1:
                            # xshift is the shift due to other arrays on CCD
                            # full_xshift is total shift when tile_all=True
                            crpix1 += xshift
                            if tile_all:
                                crpix1 += chip_xshift
                            ext_to_add.hdr.set('CRPIX1', crpix1,
                                           self.keyword_comments['CRPIX1'])
                        adoutput.append(ext_to_add[0].nddata, reset_ver=True)
                        #out_hdulist.extend(ext_to_add.to_hdulist()[1:])

                # Create new AD object, reset the EXTVERs
                #adoutput = astrodata.open(out_hdulist)
                adoutput.filename = ad.filename
                #for extver, ext in enumerate(adoutput, start=1):
                #    ext.hdr['EXTVER'] = extver

                # Update and attach OBJCAT if needed
                if any(hasattr(ext, 'OBJCAT') for ext in ad):
                    # Create new mapping as all input extensions => output 1
                    if tile_all:
                        ccd_map = np.full_like(ccd_map, 1)
                    adoutput = gt.tile_objcat(adinput=ad, adoutput=adoutput,
                                              ext_mapping=ccd_map,
                                              sx_dict=self.sx_dict)

                # Attach MDF if it exists
                if hasattr(ad, 'MDF'):
                    adoutput.MDF = ad.MDF

            # Timestamp and update filename
            gt.mark_history(adoutput, primname=self.myself(),
                            keyword=timestamp_key)
            adoutput.filename = gt.filename_updater(adoutput,
                                    suffix=params["suffix"], strip=True)
            adoutputs.append(adoutput)
        return adoutputs

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

##############################################################################
# Below are the helper functions for the primitives in this module           #
##############################################################################

def _obtain_arraygap(adinput=None):
    """
    This function obtains the raw array gap size for the different GMOS
    detectors and returns it after correcting for binning.
    """
    det_type = adinput.phu.get('DETTYPE')

    # Obtain the array gap value and fix for any binning
    arraygap = int(gmosArrayGaps[det_type] / adinput.detector_x_bin())
    return arraygap

