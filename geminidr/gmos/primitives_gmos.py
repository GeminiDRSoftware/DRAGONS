import astrodata
import gemini_instruments
from gempy.gemini import gemini_tools as gt

import numpy as np
from astropy.wcs import WCS
from astropy.table import vstack, Table, Column
from copy import deepcopy

from geminidr.core import CCD
from geminidr.gemini.primitives_gemini import Gemini
from .parameters_gmos import ParametersGMOS
from .lookups.array_gaps import gmosArrayGaps

from gempy.scripts.gmoss_fix_headers import correct_headers

from gemini_instruments.gmos.pixel_functions import get_bias_level
from gempy.gemini import eti

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

    def __init__(self, adinputs, context, ucals=None, uparms=None):
        super(GMOS, self).__init__(adinputs, context, ucals=ucals,
                                         uparms=uparms)
        self.parameters = ParametersGMOS

    def mosaicDetectors(self, adinputs=None, stream='main', **params):
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
        pars = getattr(self.parameters, self.myself())

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
                avg_overscan = np.mean(overscan for overscan in
                                ad.hdr.get('OVERSCAN') if overscan is not None)
            except TypeError:
                avg_overscan = None
            all_ampname = ','.join(ampname for ampname in ad.hdr.get('AMPNAME')
                                   if ampname is not None)

            old_detsec = min(ad.detector_section(), key=lambda x: x.x1)

            # Instantiate ETI and then run the task
            gmosaic_task = eti.gmosaiceti.GmosaicETI([], pars, ad)
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
                ad_out.hdr.set('BUNIT', bunit.pop(),
                               self.keyword_comments["BUNIT"])
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

            # TODO: Old code complained about mask being float32 and having
            # negative values that needed fixing. Doesn't seem to be true

            gt.mark_history(ad_out, primname=self.myself(), keyword=timestamp_key)
            adoutputs.append(ad_out)
        return adoutputs

    def standardizeInstrumentHeaders(self, adinputs=None, stream='main', **params):
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
        pars = getattr(self.parameters, self.myself())

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
            if ad.detector_name(pretty=True) == "Hamamatsu":
                log.status("Fixing headers for Hamamatsu data")
                #TODO: Check that this works!
                # Image extension headers appear to be correct - MS 2014-10-01
                #     correct_image_extensions=Flase
                # As does the DATE-OBS but as this seemed to break even after
                # apparently being fixed, still perform this check. - MS
                hdulist = ad.to_hdulist()
                correct_headers(hdulist, logger=log,
                                correct_image_extensions=False)
                ad = astrodata.open(hdulist)

            # Update keywords in the image extensions. The descriptors return
            # the true values on unprepared data.
            descriptors = ['pixel_scale', 'read_noise', 'gain_setting',
                               'gain', 'saturation_level']
            if 'SPECT' in ad.tags:
                descriptors.append('dispersion_axis')
            for desc in descriptors:
                keyword = ad._keyword_for(desc)
                comment = self.keyword_comments[keyword]
                dv = getattr(ad, desc)()
                if isinstance(dv, list):
                    for ext, value in zip(ad, dv):
                        ext.hdr.set(keyword, value, comment)
                else:
                    ad.hdr.set(keyword, dv, comment)

            # And the bias level too!
            bias_level = get_bias_level(adinput=ad,
                                        estimate='qa' in self.context)
            for ext, bias in zip(ad, bias_level):
                ext.hdr.set('RAWBIAS', bias, self.keyword_comments['RAWBIAS'])
            
            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=pars["suffix"],
                                              strip=True)
            adoutputs.append(ad)
        return adoutputs
    
    def standardizeStructure(self, adinputs=None, stream='main', **params):
        """
        This primitive is used to standardize the structure of GMOS data,
        specifically.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        attach_mdf: bool
            attach an MDF to the AD objects? (ignored if not tagged as SPECT)
        mdf: str
            full path of the MDF to attach
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        pars = getattr(self.parameters, self.myself())

        adoutputs = []
        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by standardizeStructure".
                            format(ad.filename))
                adoutputs.append(ad)
                continue
            
            # Attach an MDF to each input AstroData object
            if pars["attach_mdf"]:
                ad = self.addMDF(ad, mdf=pars["mdf"])

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=pars["suffix"],
                                              strip=True)
            adoutputs.append(ad)
        return adoutputs

    def tileArrays(self, adinputs=None, stream='main', **params):
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
        pars = getattr(self.parameters, self.myself())
        tile_all = pars["tile_all"]

        adoutputs = []
        for ad in adinputs:
            # Start building output AD object with the input PHU
            out_hdulist = ad.to_hdulist()[:1]

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
            ad = gt.trim_to_data_section(ad,
                                         keyword_comments=self.keyword_comments)
            new_shape = [ext.data.shape for ext in ad]
            changed = old_shape!=new_shape

            # Make chip gaps to tile with science extensions if tiling all
            # Gap width comes from a lookup table
            gap_height = int(ad[0].data.shape[0])
            gap_width = _obtain_arraygap(ad)
            chip_gap = np.zeros((gap_height,gap_width))

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

                ccd_map = np.array(ccd_map)
                for ccd in range(1, num_ccd+1):
                    amps_on_ccd = ampsorder[ccd_map==ccd]
                    extns = [ad[i] for i in amps_on_ccd]
                    # Use the centre-left amplifier's HDU as basis for new HDU
                    ref_ext = amps_on_ccd[int(len(amps_on_ccd)/2-1)]
                    # Stack the data, etc.
                    data = np.hstack([ext.data for ext in extns])
                    mask = None if any(ext.mask is None for ext in extns) \
                        else np.hstack([ext.mask for ext in extns])
                    var = None if any(ext.variance is None for ext in extns) \
                        else np.hstack([ext.variance for ext in extns])

                    # Store this information from the leftmost extension
                    if ccd==1 or not tile_all:
                        old_detsec = extns[0].detector_section()
                        old_ccdsec = extns[0].array_section()

                    # Add the widths of all arrays to the left of the reference
                    xshift = sum(ext.data.shape[1] for
                                 ext in extns[:int((len(amps_on_ccd) - 1) / 2)])

                    if tile_all and ccd>1:
                        # Set reference extension to be the centre-left of all
                        ref_ext = ampsorder[int(len(ampsorder)/2-1)]
                        # Calculate total horizontal shift if the reference
                        # array is on this CCD
                        if ref_ext in amps_on_ccd:
                            xshift += all_data.shape[1] + chip_gap.shape[1]

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
                    else:
                        all_data = data
                        all_mask = mask
                        all_var = var

                    if ccd==num_ccd or not tile_all:
                        # Append what we've got. Base it on the reference extn
                        ext_to_add = deepcopy(ad[ref_ext])
                        ext_to_add[0].reset(all_data, all_mask, all_var)

                        # Update keywords in the header
                        ext_to_add.hdr.set('CCDNAME', ad.detector_name(),
                                           self.keyword_comments['CCDNAME'])

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
                            crpix1 += xshift
                            ext_to_add.hdr.set('CRPIX1', crpix1,
                                           self.keyword_comments['CRPIX1'])

                        out_hdulist.extend(ext_to_add.to_hdulist()[1:])

                # Create new AD object, reset the EXTVERs
                adoutput = astrodata.open(out_hdulist)
                adoutput.filename = ad.filename
                for extver, ext in enumerate(adoutput, start=1):
                    ext.hdr.EXTVER = extver

                # Update and attach OBJCAT if needed
                if any(hasattr(ext, 'OBJCAT') for ext in ad):
                    # Create new mapping as all input extensions => output 1
                    if tile_all:
                        ccd_map = np.full_like(ccd_map, 1)
                    adoutput = gt.tile_objcat(adinput=ad, adoutput=adoutput,
                                              ext_mapping=ccd_map,
                                              sx_dict=self.sx_default_dict)
                    
                # Attach MDF if it exists
                if hasattr(ad, 'MDF'):
                    adoutput.MDF = ad.MDF
            
            # Timestamp and update filename
            gt.mark_history(adoutput, primname=self.myself(),
                            keyword=timestamp_key)
            adoutput.filename = gt.filename_updater(adoutput,
                                    suffix=pars["suffix"], strip=True)
            adoutputs.append(adoutput)
        
        return adoutputs

    def validateData(self, adinputs=None, stream='main', **params):
        """
        This primitive is used to validate GMOS data, specifically. The input
        AstroData object(s) are validated by ensuring that 1, 2, 3, 4, 6 or 12
        extensions are present.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        repair: bool
            Repair the data, if necessary? This does not work yet!
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        pars = getattr(self.parameters, self.myself())
        
        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by validateData".
                            format(ad.filename))
                continue
            
            # Issue a warning if the data is an image with non-square binning
            if {'GMOS', 'IMAGE'}.issubset(ad.tags):
                xbin = ad.detector_x_bin()
                ybin = ad.detector_y_bin()
                if xbin != ybin:
                    log.warning("Image {} is {} x {} binned data".
                                format(ad.filename, xbin, ybin))

            repair = pars["repair"]
            if repair:
                # Set repair to False, since it doesn't work at the moment
                log.warning("Setting repair=False, since this functionality "
                            "is not yet implemented")

            # Validate the input AstroData object by ensuring that it has
            # 1, 2, 3, 4, 6 or 12 extensions
            valid_num_ext = [1, 2, 3, 4, 6, 12]
            num_ext = len(ad)
            if num_ext not in valid_num_ext:
                if repair:
                    # This would be where we would attempt to repair the data
                    # This shouldn't happen while repair = False exists above
                    pass
                else:
                    raise IOError("The number of extensions in {} does not "
                                "match the number of extensions expected "
                                "in raw GMOS data.".format(ad.filename))
            else:
                log.fullinfo("The GMOS input file has been validated: {} "
                             "contains {} extensions".format(ad.filename, num_ext))
            
            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=pars["suffix"],
                                              strip=True)
        return adinputs

##############################################################################
# Below are the helper functions for the primitives in this module           #
##############################################################################

def _obtain_arraygap(adinput=None):
    """
    This function obtains the raw array gap size for the different GMOS
    detectors and returns it after correcting for binning.
    """
    det_type = adinput.phu.DETTYPE
    
    # Obtain the array gap value and fix for any binning
    arraygap = int(gmosArrayGaps[det_type] / adinput.detector_x_bin())
    return arraygap

