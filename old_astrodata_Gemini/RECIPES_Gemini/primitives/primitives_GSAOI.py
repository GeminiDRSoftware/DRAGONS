import numpy as np
import re
import math

from astrodata import AstroData
from astrodata.utils import logutils
from astrodata.utils.gemconstants import SCI, VAR, DQ, MDF
from astrodata.interface.slices import pixel_exts
from gempy.gemini import gemini_tools as gt
from gempy.library import mosaic as mo
from astropy import table as table
import pyfits as pf

from primitives_GEMINI import GEMINIPrimitives


class GSAOIPrimitives(GEMINIPrimitives):
    """
    This is the class containing all of the primitives for the GSAOI
    level of the type hierarchy tree. It inherits all the primitives from the
    level above, 'GEMINIPrimitives'.
    """
    astrotype = "GSAOI"
    
    def init(self, rc):
        GEMINIPrimitives.init(self, rc)
        return rc
    
    def standardizeInstrumentHeaders(self, rc):
        """
        This primitive is used to make the changes and additions to the
        keywords in the headers of GSAOI data, specifically.
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "standardizeInstrumentHeaders",
                                 "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["standardizeInstrumentHeaders"]
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the standardizeInstrumentHeaders primitive has been
            # run previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has "
                            "already been processed by "
                            "standardizeInstrumentHeaders" % ad.filename)
                
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Standardize the headers of the input AstroData object. Update the
            # keywords in the headers that are specific to GSAOI
            log.status("Updating keywords that are specific to GSAOI")
            
            # Filter name (required for IRAF?)
            gt.update_key_from_descriptor(
              adinput=ad, descriptor="filter_name(stripID=True, pretty=True)",
              keyword="FILTER", extname="PHU",
              keyword_comments=self.keyword_comments)
            
            # Pixel scale
            gt.update_key_from_descriptor(
              adinput=ad, descriptor="pixel_scale()", extname="PHU",
              keyword_comments=self.keyword_comments)
            
            # Read noise (new keyword, should it be written?)
            gt.update_key_from_descriptor(
              adinput=ad, descriptor="read_noise()", extname="SCI",
              keyword_comments=self.keyword_comments)
            
            # Gain (new keyword, should it be written?)
            gt.update_key_from_descriptor(
              adinput=ad, descriptor="gain()", extname="SCI",
              keyword_comments=self.keyword_comments)
            
            # Non linear level
            gt.update_key_from_descriptor(
              adinput=ad, descriptor="non_linear_level()", extname="SCI",
              keyword_comments=self.keyword_comments)
            
            # Saturation level
            gt.update_key_from_descriptor(
              adinput=ad, descriptor="saturation_level()", extname="SCI",
              keyword_comments=self.keyword_comments)
            
            # Dispersion axis (new keyword, should it be written?)
            if "SPECT" in ad.types:
                gt.update_key_from_descriptor(
                  adinput=ad, descriptor="dispersion_axis()", extname="SCI",
                  keyword_comments=self.keyword_comments)
            
            # Convention seems to be to multiply the exposure time by coadds
            # in prepared data
            # DON'T DO THIS: GSAOI data are averaged over coadds, not summed
            #gt.update_key_from_descriptor(
            #  adinput=ad, descriptor="exposure_time()", extname="PHU")

            # Adjust CCDSEC to deal with a bug
            date = ad.phu_get_key_value('DATE-OBS')
            for sciext in ad["SCI"]:

                # There is a bug in GSAOI data where the central 1K x 1K ROI
                # only is read out (usually for photometric standards) and the
                # CCDSEC keyword needs to be fixed for this type of data so the
                # auxiliary data (i.e. BPM and flats) match. Best guess date
                # for start of problem is currently May 15 2013. It is not
                # currently clear if this bug is in the detector controller
                # code or the SDSU board timing.
                secstr = str(sciext.array_section(pretty=True))
                if secstr == '[513:1536,513:1536]' and date >= '2013-05-15':
                    log.stdinfo("Updating the CCDSEC for central ROI data")
                    extver = sciext.extver()
                    y1o = 513 if extver < 3 else 511
                    y2o = y1o + 1024
                    secstr = "[{0}:{1},{2}:{3}]".format(513, 1536, y1o+1, y2o)
                    sciext.set_key_value('CCDSEC', secstr)
                
                                    
            # Move BUNIT keyword from the primary header, where GSAOI dubiously
            # puts it, to the SCI extensions, where the descriptors and
            # ADUToElectrons expect to find/put it:
            bunit = ad.phu_get_key_value('BUNIT')
            if bunit is not None:
                del ad.phu.header['BUNIT']  # there's no more-idiomatic way...
                for sciext in ad["SCI"]:
                    sciext.set_key_value('BUNIT', bunit,
                                         'Units of the array values')

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)
            
            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)
            
            # Append the output AstroData object to the list of output
            # AstroData objects 
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction context
        rc.report_output(adoutput_list)
        
        yield rc 
    
    def standardizeStructure(self, rc):
        """
        This primitive is used to standardize the structure of GSAOI
        data, specifically.
        
        :param attach_mdf: Set to True to attach an MDF extension to the input
                           AstroData object(s). If an input AstroData object
                           does not have an AstroData type of SPECT, no MDF
                           will be added, regardless of the value of this
                           parameter.
        :type attach_mdf: Python boolean
        :param mdf: The file name, including the full path, of the MDF(s) to
                    attach to the input AstroData object(s). If only one MDF is
                    provided, that MDF will be attached to all input AstroData
                    object(s). If more than one MDF is provided, the number of
                    MDFs must match the number of input AstroData objects. If
                    no MDF is provided, the primitive will attempt to determine
                    an appropriate MDF.
        :type mdf: string or list of strings
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "standardizeStructure",
                                 "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["standardizeStructure"]
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Use a flag to determine whether to run addMDF
        attach_mdf = True
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the standardizeStructure primitive has been run
            # previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has "
                            "already been processed by standardizeStructure"
                            % ad.filename)
                
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Attach an MDF to each input AstroData object
            if rc["attach_mdf"] and attach_mdf:
                
                # Get the mdf parameter from the reduction context
                mdf = rc["mdf"]
                if mdf is not None:
                    rc.run("addMDF(mdf=%s)" % mdf)
                else:
                    rc.run("addMDF")
                
                # Since addMDF uses all the AstroData inputs from the reduction
                # context, it only needs to be run once in this loop
                attach_mdf = False
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)
            
            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)
            
            # Append the output AstroData object to the list of output
            # AstroData objects 
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction context
        rc.report_output(adoutput_list)
        
        yield rc

    def tileArrays(self,rc):
        
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "tileArrays", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["tileArrays"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():

            # This RC parameter seems to determine whether to tile only the
            # constituent amps of each detector or all the detectors (+ their
            # gaps) as well, in place of later mosaicking. Since each GSAOI
            # detector is always read out as a single extension, setting
            # tile_all to False makes this primitive a no-op.
            tile_all = rc["tile_all"]

            # For GSAOI, we'll simply omit any VAR & DQ extensions, since
            # tiling is done for display or analysis purposes only and does
            # not go into the final images.
            # - Probably need to restore DQ for flagging bad pixels though.
            #   - Sorry, there's probably a bit of bookeeping to factor out,
            #     to avoid cutting and pasting when adding that.

            # Store PHU to pass to output AD
            # The original name must be stored first so that the
            # output AD can reconstruct it later
            ad.store_original_name()
            phu = ad.phu

            # Do nothing if there is only one science extension:
            nsciext = ad.count_exts("SCI")
            if not tile_all or nsciext == 1:
                log.fullinfo("Only one science extension found or tile_all "\
                             "disabled;\n no tiling done for %s" % ad.filename)
                adoutput = ad

            # Otherwise, trim & tile the data:
            else:
                # Ensure extensions are accessed in strict numerical order
                # So VAR and DQ planes match SCI planes
                sci_exts = [ad['SCI',ext] for ext in range(1,nsciext+1)]
                
                if len(ad) > 0:
                    log.fullinfo("Tiling all data into one extension "\
                                 "(without VAR/DQ)")
                else:
                    raise ValueError(
                        'input "{0}" has no SCI data'.format(ad.filename)
                    )

                # Make a new AD instance for the output:
                adoutput = AstroData()
                adoutput.filename = ad.filename
                adoutput.phu = phu
        
                # First trim off any unused border regions still present
                # so they won't get tiled with science data:
                log.fullinfo("Trimming to data section:")
                ad = gt.trim_to_data_section(adinput=ad, keyword_comments=self.keyword_comments)[0]

                # Check that the SCI,VAR,DQ planes are all the same size.
                # If they're not, something funky has happened, but it would
                # be useful to raise a suitable error
                sizes_match = True
                for ext in range(1,nsciext+1):
                    if ad[VAR,ext] is not None:
                        sizes_match &= (ad[VAR,ext].data.shape ==
                                        ad[SCI,ext].data.shape)
                    if ad[DQ,ext] is not None:
                        sizes_match &= (ad[DQ,ext].data.shape ==
                                        ad[SCI,ext].data.shape)
                    if ad['OBJMASK',ext] is not None:
                        sizes_match &= (ad['OBJMASK',ext].data.shape ==
                                        ad[SCI,ext].data.shape)
                if not sizes_match:
                    raise Errors.InputError("Planes have different shapes")

                # The following mosaicking method is shamelessly adapted from
                # gempylocal dataset.tile.OnSkyCheckImg.tile(), which we can't
                # depend on, as it lives in a class in gemaux (yes, it ends up
                # being a fair amount of work just to paste 4 arrays together):

                # Cache the detector sections & find the detector coordinates
                # of the bottom-left corner of the central array:
                detsec_coord = [tuple(ext.detector_section().as_pytype()) \
                                for ext in sci_exts]

                # This has to be calculated here because we're changing
                # detsec_coord
                detsec = mo.combine_limits(detsec_coord, to_FITS=True)

                # From GSAOI Web page, the gaps between detectors are ~2.5".
                # This needs moving into some sort of look-up if we ever plan
                # to make this primitive more generic:
                gap_size, detsec_coord = _squeeze_detectors(
                            2.5 / ad.pixel_scale().as_float(),
                            detsec_coord)

                # Reset co-ordinate origins relative to the set being tiled:
                mosaic_coord = mo.reset_origins(detsec_coord)
                block_coord = mo.reset_origins(detsec_coord, per_block=True)

                # Construct array position information needed by mosaic:
                tile_coords = {
                    'amp_mosaic_coord' : mosaic_coord,
                    'amp_block_coord' : block_coord
                }

                # Since each tuple in block_coord has its origin reset to zero,
                # the array sizes can be retrieved from every second element:
                sizes = set(coord_set[1::2] for coord_set in block_coord)
                if len(sizes) == 1:
                    size, = sizes
                elif sizes:
                    raise ValueError('equal sub-array sizes are required')

                # The case of a single-quadrant ROI, if used, would be handled
                # by the above condition that skips tiling. Treat everything
                # else as a 2x2 grid, since anything in between would just be
                # an ambiguous subset of that anyway -- and Rodrigo tells me
                # that, in practice, all 4 quadrants are read whatever the ROI.

                mosaic_geometry = mo.MosaicGeometry(
                    {'blocksize' : size,
                     'mosaic_grid' : (2,2),
                     'gaps' : {(y, x) : (gap_size, gap_size) \
                               for x in (0,1) for y in (0,1)}
                    }
                )
                
                nextend = 0
                padding_mask = None
                for ext_type in [SCI,VAR,DQ,'OBJMASK']:
                    if ad[ext_type] is not None:
                        data_list = [ext.data for ext in [ad[ext_type,i]
                                                for i in range(1,nsciext+1)]]
                        # The pixel data and the tile coordinates are used
                        # to build a MosaicData object.
                        mosaic_data = mo.MosaicData(data_list, tile_coords)

                        # Use the MosaicData object to build a Mosaic object
                        mosaic = mo.Mosaic(mosaic_data,
                                           mosaic_geometry=mosaic_geometry)

                        # Actually create the tiled pixel data array.
                        tiled_data = mosaic.mosaic_image_data()

                        # Use the SCI Mosaic mask to determine what's data
                        if ext_type == SCI:
                            padding_mask = mosaic.mask
                        elif ext_type == DQ and padding_mask is not None:
                            # DQ=16 means "no data"
                            tiled_data |= padding_mask*16                            

                        # Append result to output, re-using first input header
                        # (appears to make a copy of the header automatically)
                        adoutput.append(AstroData(data=tiled_data,
                                header=ad[ext_type,1].header),
                                extname=ext_type, extver=1)
                        nextend += 1

                # Update the zero point of the WCS inherited from the first
                # extension. First derive the extension's zero points within
                # the combined mosaic. I'm not sure whether there's a
                # fractional-pixel error WRT what mosaic does here but we're
                # only "tiling" anyway, not making an accurate mosaic:
                # CJS: Compute offsets for all extensions to aid in OBJCAT
                # concatenation later (WCS fix will only use cr_off[0])
                cr_off = []
                for i in range(nsciext):
                    cr_off.append(
                        tuple((llim+gap_size if llim >= arr_size else llim) \
                        for arr_size, llim in \
                            zip(ad[SCI,i+1].data.shape, mosaic_coord[i][::2])))
 
                # Iterate over SCI, VAR, DQ
                for ext in adoutput:
                    # These keywords are no longer applicable after tiling:
                    for keyword in ('CCDNAME', 'TRIMSEC'):
                        del ext.header[keyword]

                    # Update other section keywords:
                    shape = ext.data.shape
                    ext.set_key_value(
                        'DATASEC',
                        '[1:{0},1:{1}]'.format(*ext.data.shape),
                        comment=self.keyword_comments['DATASEC']
                        )
                    ext.set_key_value(
                        'CCDSEC',
                        '[1:{0},1:{1}]'.format(*ext.data.shape),
                        comment=self.keyword_comments['DATASEC']
                        )
                
                    # Record a combined DETSEC for the extensions. This doesn't
                    # actually match the array dimensions, due to the gaps, but
                    # still represents the range of the full, contiguous detector
                    # mosaic that is spanned by the data.
                    ext.set_key_value(
                        'DETSEC',
                        '[{0}:{1},{2}:{3}]'.format(*detsec),
                        comment=self.keyword_comments['DETSEC']
                        )

                    # Apply the appropriate offset to each CRPIXN keyword:
                    for n, off in enumerate(cr_off[0], start=1):
                        key = 'CRPIX{0}'.format(n)
                        crpix = ext.get_key_value(key)
                        if crpix is not None:
                            ext.set_key_value(
                                key, float(crpix) + off,
                                comment=self.keyword_comments[key]
                            )
                            
                # And attach a concatenated OBJCAT
                if ad['OBJCAT'] is not None:
                    adoutput = _tile_objcat(ad,adoutput,cr_off)
                    nextend += 1
                    
                # Attach one of the REFCATs if they're there
                # (all REFCATs should be the same)
                if ad['REFCAT'] is not None:
                    adoutput.append(ad['REFCAT',1])
                    nextend += 1

                # Update Gemini header bookkeeping as for GMOS:
                adoutput.phu_set_key_value(
                    "NSCIEXT", 1, comment=self.keyword_comments["NSCIEXT"])
                adoutput.phu_set_key_value(
                    "NEXTEND", nextend,
                    comment=self.keyword_comments["NEXTEND"])

            # Refresh AstroData types in output file (original ones
            # were lost when new AD was created)
            adoutput.refresh_types()

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=adoutput, primname=self.myself(), keyword=timestamp_key)

            # Change the filename
            adoutput.filename = gt.filename_updater(
                adinput=adoutput, suffix=rc["suffix"], strip=True)
            
            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(adoutput)
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc

    def validateData(self, rc):
        """
        This primitive is used to validate GSAOI data, specifically.
        """
        
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "validateData", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["validateData"]
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the validateData primitive has been run previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has "
                            "already been processed by validateData"
                            % ad.filename)
                
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Validate the input AstroData object by ensuring that it has
            # 4 science extensions (NOTE: this might not be correct for 
            # spectroscopy)
            num_ext = ad.count_exts("SCI")
            if num_ext != 4:
                raise Errors.Error("The number of extensions in %s do "
                                    "match with the number of extensions "
                                    "expected in raw GSAOI data."
                                    % ad.filename)
            else:
                log.fullinfo("The GSAOI input file has been validated: %s "
                             "contains %d extensions" % (ad.filename, num_ext))
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)
            
            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)
            
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction context
        rc.report_output(adoutput_list)
        
        yield rc
        
##############################################################################
# Below are the helper functions for the primitives in this module           #
##############################################################################

def _squeeze_detectors(default_gap, detsec_coord):
    # DETSEC blocks should be contiguous. If they're not, make them so and
    # increase the gap size appropriately (for display purposes).
    # This is a massive fudge caused in part by philosophical differences

    gap_size = default_gap

    # Determine bottom left of coords entire shebang (will reset to 0 later)
    xmin = min(dc[0] for dc in detsec_coord)
    ymin = min(dc[2] for dc in detsec_coord)
    
    # Do x first, then y
    for index in [0,2]:
        # Take the bottom-left detector, work out how far in x it extends
        # Then find the detector with the lowest x that's to the right of this
        # If there's a gap, close it up and change gap_size accordingly
        thismin = xmin if index==0 else ymin
        end = max(dc[index+1] for dc in detsec_coord if dc[index]==thismin)
        squeezable = [dc[index] for dc in detsec_coord if dc[index]>=end]
        while len(squeezable)>0:
            this_gap = min(squeezable) - end
            if this_gap > 0:
                for i in range(len(detsec_coord)):
                    if detsec_coord[i][index] > end:
                        dc = detsec_coord[i]
                        if index == 0:
                            # Each detsec element is a tuple, so can't edit
                            # single elements
                            detsec_coord[i] = (dc[0]-this_gap, dc[1]-this_gap,
                                           dc[2], dc[3])
                        else:
                            detsec_coord[i] = (dc[0], dc[1],
                                           dc[2]-this_gap, dc[3]-this_gap)
                gap_size = max(gap_size, default_gap+this_gap)
            end = max(dc[index+1] for dc in detsec_coord if dc[index]==end)
            squeezable = [dc[index] for dc in detsec_coord if dc[index]>=end]

    # Reset bottom-left to zero
    for i in range(len(detsec_coord)):
        dc = detsec_coord[i]
        detsec_coord[i] = (dc[0]-xmin, dc[1]-xmin,
                           dc[2]-ymin, dc[3]-ymin)
    
    return gap_size, detsec_coord

def _tile_objcat(adinput,adoutput,cr_off=None):
    
    if adinput['OBJCAT'] is None:
        raise Errors.InputError("No OBJCAT found in %s" % ad.filename)
    else:
        # Go through the extensions. Keep hold of the first OBJCAT we find
        # to use its column information
        first_objcat = None

        for extver in range(1,len(adinput['SCI'])+1):
            inp_objcat = adinput['OBJCAT',extver]
            if inp_objcat is None:
                continue
            inp_table = table.Table(inp_objcat.data)
            if cr_off is not None:
                inp_table['X_IMAGE'] += int(cr_off[extver-1][0])
                inp_table['Y_IMAGE'] += int(cr_off[extver-1][1])
            # Make new output table, or append to existing output table
            if first_objcat is None:
                out_table = inp_table
                first_objcat = inp_objcat.data.columns
            else:
                inp_table['NUMBER'] += len(out_table)
                out_table = table.vstack([out_table,inp_table])
        
        # CJS: Clunky: astropy.table doesn't allow writing to MEFs (yet),
        # so we need to parse this through pyFITS
        table_columns = [c for c in first_objcat]
        for c in table_columns:
            c.array = out_table[c.name]
        col_def = pf.ColDefs(table_columns)
        tb_hdu  = pf.new_table(col_def)
        tb_ad   = AstroData(tb_hdu)
        tb_ad.rename_ext("OBJCAT", 1)
        adoutput.append(tb_ad)
        del first_objcat
        
    return adoutput
