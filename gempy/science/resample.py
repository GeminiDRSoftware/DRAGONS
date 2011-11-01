# This module contains user level functions related to resampling the input
# dataset

import os
import sys
import numpy as np
import pyfits as pf
import pywcs
from astrodata import AstroData
from astrodata import Errors
from astrodata import Lookups
from astrodata.adutils import gemLog
from astrodata.adutils.gemutil import pyrafLoader
from gempy import astrotools as at
from gempy import geminiTools as gt
from gempy import managers as mgr
from gempy.geminiCLParDicts import CLDefaultParamsDict
from gempy.gemini_metadata_utils import sectionStrToIntList
from gempy.science.preprocessing import bias as bs

# Load the timestamp keyword dictionary that will be used to define the keyword
# to be used for the time stamp for the user level function
timestamp_keys = Lookups.get_lookup_table("Gemini/timestamp_keywords",
                                          "timestamp_keys")

def align_to_reference_image(adinput, interpolator="linear", trim_data=False):
    """
    This function applies the transformation encoded in the input images
    WCSs to align them with a reference image, in reference image pixel
    coordinates. The reference image is taken to be the first image in
    the input list.
    
    By default, the transformation into the reference frame is done via
    interpolation. The interpolator parameter specifies the interpolation 
    method. The options are nearest-neighbor, bilinear, or nth-order 
    spline, with n = 2, 3, 4, or 5. If interpolator is None, 
    no interpolation is done: the input image is shifted by an integer
    number of pixels, such that the center of the frame matches up as
    well as possible. The variance plane, if present, is transformed in
    the same way as the science data.
    
    The data quality plane, if present, must be handled a little
    differently. DQ flags are set bit-wise, such that each pixel is the 
    sum of any of the following values: 0=good pixel,
    1=bad pixel (from bad pixel mask), 2=nonlinear, 4=saturated, etc.
    To transform the DQ plane without losing flag information, it is
    unpacked into separate masks, each of which is transformed in the same
    way as the science data. A pixel is flagged if it had greater than
    1% influence from a bad pixel. The transformed masks are then added
    back together to generate the transformed DQ plane.
    
    In order not to lose any data, the output image arrays (including the
    reference image's) are expanded with respect to the input image arrays.
    The science and variance data arrays are padded with zeros; the DQ
    plane is padded with ones.
    
    The WCS keywords in the headers of the output images are updated
    to reflect the transformation.
    
    :param adinput: list of images to align. First image is taken to be
                  the reference image.
    :type adinput: list of AstroData objects
    
    :param interpolator: type of interpolation desired
    :type interpolator: string, possible values are None, 'nearest', 'linear',
                        'spline2', 'spline3', 'spline4', or 'spline5'

    :param trim_data: flag to indicate whether output image should be trimmed
                      to the size of the reference image.
    :type trim_data: Boolean
    """
    
    # instantiate log
    log = gemLog.getGeminiLog()
    
    # Ensure that adinput is not None and return
    # a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)
    
    # keyword to be used for time stamp
    timestamp_key = timestamp_keys["align_to_reference_image"]
    
    adoutput_list = []
    try:
        # check for at least two input images (first one is reference)
        if len(adinput)<2:
            raise Errors.InputError("At least two input images " +
                                    "must be supplied.")
        
        # make sure all images have one science extension
        for ad in adinput:
            sci_exts = ad["SCI"]
            if sci_exts is None or len(sci_exts)!=1:
                raise Errors.InputError("Input images must have only one " +
                                        "SCI extension.")
        
        # load ndimage package if there will be interpolation
        if interpolator=="None":
            interpolator = None
        if interpolator is not None:
            from scipy.ndimage import affine_transform
        
        # get reference WCS
        reference = adinput[0]
        ref_wcs = pywcs.WCS(reference["SCI"].header)
        ref_shape = reference["SCI"].data.shape
        ref_corners = at.get_corners(ref_shape)
        naxis = len(ref_shape)
        
        # first pass: get output image shape required to fit all data in output
        # by transforming corner coordinates of images
        all_corners = [ref_corners]
        xy_img_corners = []
        shifts = []
        for i in range(1,len(adinput)):
            
            ad = adinput[i]
            
            img_wcs = pywcs.WCS(ad["SCI"].header)
            
            img_shape = ad["SCI"].data.shape
            img_corners = at.get_corners(img_shape)
            
            xy_corners = [(corner[1],corner[0]) for corner in img_corners]
            xy_img_corners.append(xy_corners)
            
            if interpolator is None:
                # find shift by transforming center position of field
                # (so that center matches best)
                x1y1 = np.array([img_shape[1]/2.0,img_shape[0]/2.0])
                x2y2 = img_wcs.wcs_sky2pix(ref_wcs.wcs_pix2sky([x1y1],1),1)[0]
                
                # round shift to nearest integer and flip x and y
                offset = np.roll(np.rint(x2y2-x1y1),1)
                
                # shift corners of image
                img_corners = [tuple(offset+corner) 
                               for corner in img_corners]
                shifts.append(offset)
            else:
                # transform corners of image via WCS
                xy_corners = img_wcs.wcs_sky2pix(ref_wcs.wcs_pix2sky(
                                                           xy_corners,0),0)
                
                img_corners = [(corner[1],corner[0]) for corner in xy_corners]
            
            all_corners.append(img_corners)
        
        # If data should be trimmed to size of reference image,
        # output shape is same as ref_shape, and centering offsets are zero
        if trim_data:
            cenoff=[0]*naxis
            out_shape = ref_shape
        else:
            # Otherwise, use the corners of the images to get the minimum
            # required output shape to hold all data
            cenoff = []
            out_shape = []
            for axis in range(naxis):
                # get output shape from corner values
                cvals = [corner[axis] for ic in all_corners for corner in ic]
                out_shape.append(int(max(cvals)-min(cvals)+1))
                
                # if just shifting, need to set centering shift for reference
                # image from offsets already calculated
                if interpolator is None:
                    svals = [shift[axis] for shift in shifts]
                    # include a 0 shift for the reference image
                    # (in case it's already centered)
                    svals.append(0.0)
                    cenoff.append(-int(max(svals)))
        
            out_shape = tuple(out_shape)
        
            # if not shifting, get offset required to center reference image
            # from the size of the image
            if interpolator is not None:
                incen = [0.5*(axlen-1) for axlen in ref_shape]
                outcen = [0.5*(axlen-1) for axlen in out_shape]
                cenoff = np.rint(incen) - np.rint(outcen)

        # shift the reference image to keep it in the center of the new array
        # (do the same for VAR and DQ)

        if trim_data:
            log.fullinfo("Trimming data to size of reference image")
        else:
            log.fullinfo("Growing reference image to keep all data; " +
                         "centering data, and updating WCS to account " +
                         "for shift")
            log.fullinfo("New output shape: "+repr(out_shape))
        
        ref_corners = [(corner[1]-cenoff[1]+1,corner[0]-cenoff[0]+1) # x,y
                       for corner in ref_corners]
        log.fullinfo("Setting AREA keywords in header to denote original " +
                     "data area.")
        area_keys = []
        log.fullinfo("AREATYPE = 'P4'     / Polygon with 4 vertices")
        area_keys.append(("AREATYPE","P4","Polygon with 4 vertices"))
        for i in range(len(ref_corners)):
            for axis in range(len(ref_corners[i])):
                key_name = "AREA%i_%i" % (i+1,axis+1)
                key_value = ref_corners[i][axis]
                key_comment = "Vertex %i, dimension %i" % (i+1,axis+1)
                area_keys.append((key_name,key_value,key_comment))
                log.fullinfo("%-8s = %7.2f  / %s" % 
                             (key_name, key_value,key_comment))
        
        for ext in reference:
            if ext.extname() not in ["SCI","VAR","DQ"]:
                continue
            
            ref_data = ext.data
            
            trans_data = np.zeros(out_shape)
            
            # pad the DQ plane with 1 instead of 0
            if ext.extname()=="DQ":
                trans_data += 1.0
            
            trans_data[int(-cenoff[0]):int(ref_shape[0]-cenoff[0]),
                       int(-cenoff[1]):int(ref_shape[1]-cenoff[1])] = ref_data
            
            ext.data = trans_data
            
            # update the WCS in the reference image to account for the shift
            ext.set_key_value("CRPIX1", ref_wcs.wcs.crpix[0]-cenoff[1])
            ext.set_key_value("CRPIX2", ref_wcs.wcs.crpix[1]-cenoff[0])
            
            # set area keywords
            for key in area_keys:
                ext.set_key_value(key[0],key[1],key[2])
        
        # update the WCS in the PHU as well
        reference.phu_set_key_value("CRPIX1", ref_wcs.wcs.crpix[0]-cenoff[1])
        reference.phu_set_key_value("CRPIX2", ref_wcs.wcs.crpix[1]-cenoff[0])
        
        out_wcs = pywcs.WCS(reference["SCI"].header)
        
        adoutput_list.append(reference)
        
        # now transform the data
        for i in range(1,len(adinput)):
            
            log.fullinfo("Starting alignment for "+ adinput[i].filename)
            
            ad = adinput[i]
            
            sciext = ad["SCI"]
            img_wcs = pywcs.WCS(sciext.header)
            img_shape = sciext.data.shape
            
            if interpolator is None:
                
                # recalculate shift from new reference wcs
                x1y1 = np.array([img_shape[1]/2.0,img_shape[0]/2.0])
                x2y2 = img_wcs.wcs_sky2pix(out_wcs.wcs_pix2sky([x1y1],1),1)[0]
                
                shift = np.roll(np.rint(x2y2-x1y1),1)
                if np.any(shift>0):
                    log.warning("Shift was calculated to be >0; " +
                                "interpolator=None "+
                                "may not be appropriate for this data.")
                    shift = np.where(shift>0,0,shift)
                
                # update PHU WCS keywords
                log.fullinfo("Offsets: "+repr(np.roll(shift,1)))
                log.fullinfo("Updating WCS to track shift in data")
                ad.phu_set_key_value("CRPIX1", img_wcs.wcs.crpix[0]-shift[1])
                ad.phu_set_key_value("CRPIX2", img_wcs.wcs.crpix[1]-shift[0])
            
            else:
                # get transformation matrix from composite of wcs's
                # matrix = in_sky2pix*out_pix2sky (converts output to input)
                xy_matrix = np.dot(
                    np.linalg.inv(img_wcs.wcs.cd),out_wcs.wcs.cd)
                # switch x and y for compatibility with numpy ordering
                flip_xy = np.roll(np.eye(2),2)
                matrix = np.dot(flip_xy,np.dot(xy_matrix,flip_xy))
                matrix_det = np.linalg.det(matrix)
                
                # offsets: shift origin of transformation to the reference
                # pixel by subtracting the transformation of the output
                # reference pixel and adding the input reference pixel back in
                refcrpix = np.roll(out_wcs.wcs.crpix,1)
                imgcrpix = np.roll(img_wcs.wcs.crpix,1)
                offset = imgcrpix - np.dot(matrix,refcrpix)
                
                # then add in the shift of origin due to dithering offset.
                # This is the transform of the reference CRPIX position,
                # minus the original position
                trans_crpix = img_wcs.wcs_sky2pix(out_wcs.wcs_pix2sky(
                                                 [out_wcs.wcs.crpix],1),1)[0]
                trans_crpix = np.roll(trans_crpix,1)
                offset = offset + trans_crpix-imgcrpix
                
                # Since the transformation really is into the reference
                # WCS coordinate system as near as possible, just set image
                # WCS equal to reference WCS
                log.fullinfo("Offsets: "+repr(np.roll(offset,1)))
                log.fullinfo("Transformation matrix:\n"+repr(matrix))
                log.fullinfo("Updating WCS to match reference WCS")
                ad.phu_set_key_value("CRPIX1", out_wcs.wcs.crpix[0])
                ad.phu_set_key_value("CRPIX2", out_wcs.wcs.crpix[1])
                ad.phu_set_key_value("CRVAL1", out_wcs.wcs.crval[0])
                ad.phu_set_key_value("CRVAL2", out_wcs.wcs.crval[1])
                ad.phu_set_key_value("CD1_1", out_wcs.wcs.cd[0,0])
                ad.phu_set_key_value("CD1_2", out_wcs.wcs.cd[0,1])
                ad.phu_set_key_value("CD2_1", out_wcs.wcs.cd[1,0])
                ad.phu_set_key_value("CD2_2", out_wcs.wcs.cd[1,1])
            
            # transform corners to find new location of original data
            data_corners = out_wcs.wcs_sky2pix(img_wcs.wcs_pix2sky(
                                                  xy_img_corners[i-1],0),1)
            log.fullinfo("Setting AREA keywords in header to denote " +
                         "original data area.")
            area_keys = []
            log.fullinfo("AREATYPE = 'P4'     / Polygon with 4 vertices")
            area_keys.append(("AREATYPE","P4","Polygon with 4 vertices"))
            for i in range(len(data_corners)):
                for axis in range(len(data_corners[i])):
                    key_name = "AREA%i_%i" % (i+1,axis+1)
                    key_value = data_corners[i][axis]
                    key_comment = "Vertex %i, dimension %i" % (i+1,axis+1)
                    area_keys.append((key_name,key_value,key_comment))
                    log.fullinfo("%-8s = %7.2f  / %s" % 
                                 (key_name, key_value,key_comment))
            
            for ext in ad:
                extname = ext.extname()
                
                if extname not in ["SCI","VAR","DQ"]:
                    continue
                
                log.fullinfo("Transforming "+ad.filename+"["+extname+"]")
                
                # Access pixel data
                img_data = ext.data
                
                if interpolator is None:
                    # just shift the data by an integer number of pixels
                    # (useful for noisy data, also lightning fast)
                    
                    trans_data = np.zeros(out_shape)
                    
                    # pad the DQ plane with 1 instead of 0
                    if extname=="DQ":
                        trans_data += 1
                    
                    trans_data[int(-shift[0]):int(img_shape[0]
                                                  -shift[0]),
                               int(-shift[1]):int(img_shape[1]
                                                  -shift[1])] = img_data
                    
                    matrix_det = 1.0
                    
                    # update the wcs to track the transformation
                    ext.set_key_value("CRPIX1", img_wcs.wcs.crpix[0]-shift[1])
                    ext.set_key_value("CRPIX2", img_wcs.wcs.crpix[1]-shift[0])
                
                else:
                    # use ndimage to interpolate values
                    
                    # Interpolation method is determined by 
                    # interpolator parameter
                    if interpolator=="nearest":
                        order = 0
                    elif interpolator=="linear":
                        order = 1
                    elif interpolator=="spline2":
                        order = 2
                    elif interpolator=="spline3":
                        order = 3
                    elif interpolator=="spline4":
                        order = 4
                    elif interpolator=="spline5":
                        order = 5
                    else:
                        raise Errors.InputError("Interpolation method " +
                                                interpolator +
                                                " not recognized.")
                    
                    if extname=="DQ":
                        
                        # DQ flags are set bit-wise
                        # bit 1: bad pixel (1)
                        # bit 2: nonlinear (2)
                        # bit 3: saturated (4)
                        # A pixel can be 0 (good, no flags), or the sum of
                        # any of the above flags 
                        # (or any others I don't know about)
                        
                        # unpack the DQ data into separate masks
                        # NOTE: this method only works for 8-bit masks!
                        unp = (img_shape[0],img_shape[1],8)
                        unpack_data = np.unpackbits(np.uint8(
                                                      img_data)).reshape(unp)
                        
                        # transform each mask
                        trans_data = np.zeros(out_shape)
                        for j in range(0,8):
                            
                            # skip the transformation if there are no flags set
                            # (but always do the bad pixel mask because it is 
                            # needed to mask the part of the array that was
                            # padded out to match the reference image)
                            if not unpack_data[:,:,j].any() and j!=7:
                                # first bit is j=7 because unpack is backwards 
                                continue
                            
                            mask = np.float32(unpack_data[:,:,j])
                            
                            # if bad pix bit, pad with 1. Otherwise, pad with 0
                            if j==7:
                                cval = 1
                            else:
                                cval = 0
                            trans_mask = affine_transform(
                                mask, matrix, offset=offset,
                                output_shape=out_shape, order=order, cval=cval)
                            del mask; mask = None
                            
                            # flag any pixels with >1% influence from bad pixel
                            trans_mask = np.where(np.abs(trans_mask)>0.01,
                                                  2**(7-j),0)
                            
                            # add the flags into the overall mask
                            trans_data += trans_mask
                            del trans_mask; trans_mask = None
                    
                    else:
                        
                        # transform science and variance data in the same way
                        cval = 0.0
                        trans_data = affine_transform(img_data, matrix,
                                                      offset=offset,
                                                      output_shape=out_shape,
                                                      order=order, cval=cval)
                    
                    # update the wcs
                    ext.set_key_value("CRPIX1", out_wcs.wcs.crpix[0])
                    ext.set_key_value("CRPIX2", out_wcs.wcs.crpix[1])
                    ext.set_key_value("CRVAL1", out_wcs.wcs.crval[0])
                    ext.set_key_value("CRVAL2", out_wcs.wcs.crval[1])
                    ext.set_key_value("CD1_1", out_wcs.wcs.cd[0,0])
                    ext.set_key_value("CD1_2", out_wcs.wcs.cd[0,1])
                    ext.set_key_value("CD2_1", out_wcs.wcs.cd[1,0])
                    ext.set_key_value("CD2_2", out_wcs.wcs.cd[1,1])
                    
                    # set area keywords
                    for key in area_keys:
                        ext.set_key_value(key[0],key[1],key[2])
                
                ext.data = trans_data
            
            # if there was any scaling in the transformation, the
            # pixel size will have changed, and the output should
            # be scaled by the ratio of input pixel size to output
            # pixel size to conserve the total flux in a feature.
            # This factor is the determinant of the transformation
            # matrix.
            if (1.0-matrix_det)>1e-6:
                log.fullinfo("Multiplying by %f to conserve flux" %
                             matrix_det)
                
                # Allow the arith toolbox to do the multiplication
                # so that variance is handled correctly
                ad.mult(matrix_det)
            
            # Add time stamp to PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)
            adoutput_list.append(ad)
        
        return adoutput_list
    
    except:
        # Log the message from the exception
        log.error(repr(sys.exc_info()[1]))
        raise

def mosaic_detectors(adinput, tile=False, interpolator="linear"):
    """
    This function will mosaic the SCI frames of the input images, 
    along with the VAR and DQ frames if they exist.
    
    WARNING: The gmosaic script used here replaces the previously 
    calculated DQ frames with its own versions. This may be corrected 
    in the future by replacing the use of the gmosaic
    with a Python routine to do the frame mosaicing.
    
    NOTE: The inputs to this function MUST be prepared.
    
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
    
    :param adinput: Astrodata inputs to mosaic the extensions of
    :type adinput: Astrodata objects, either a single or a list of objects
    
    :param tile: Tile images instead of mosaic?
    :type tile: Python boolean (True/False)
    
    :param interpolator: type of interpolation algorithm to use for between 
                            the chip gaps.
    :type interpolator: string, options: 'linear', 'nearest', 'poly3', 
                           'poly5', 'spine3', 'sinc'.
    """
    
    # instantiate log
    log = gemLog.getGeminiLog()
    
    # ensure that adinput is not None and make it into a list
    # if it is not one already
    adinput = gt.validate_input(adinput=adinput)
    
    # time stamp keyword
    timestamp_key = timestamp_keys["mosaic_detectors"]
    
    # initialize output list
    adoutput_list = []
    
    try:
        # load and bring the pyraf related modules into the name-space
        pyraf, gemini, yes, no = pyrafLoader()
            
        for ad in adinput:

            # Check whether this user level function has been
            # run previously
            if ad.phu_get_key_value(timestamp_key):
                raise Errors.InputError("%s has already been processed by " \
                                        "mosaic_detectors" % (ad.filename))

            # Get BUNIT, OVERSCAN,and AMPNAME from science extensions 
            # (gmosaic wipes out these keywords, they need to 
            # be restored after runnning it)
            bunit = None
            overscan = []
            ampname = []
            for ext in ad["SCI"]:
                ext_bunit = ext.get_key_value("BUNIT")
                if bunit is None:
                    bunit = ext_bunit
                else:
                    if ext_bunit!=bunit:
                        raise Errors.ScienceError("BUNIT needs to be the" +
                                                  "same for all extensions")
                ext_overscan = ext.get_key_value("OVERSCAN")
                if ext_overscan is not None:
                    overscan.append(ext_overscan)

                ext_ampname = ext.get_key_value("AMPNAME")
                if ext_ampname is not None:
                    ampname.append(ext_ampname)

            if len(overscan)>0:
                avg_overscan = np.mean(overscan)
            else:
                avg_overscan = None

            if len(ampname)>0:
                all_ampname = ",".join(ampname)
            else:
                all_ampname = None

            # Save detector section from 1st extension
            old_detsec = ad["SCI",1].detector_section().as_list()

            # Determine whether VAR/DQ needs to be propagated
            if (ad.count_exts("VAR") == 
                ad.count_exts("DQ") == 
                ad.count_exts("SCI")):
                fl_vardq=yes
            else:
                fl_vardq=no
            
            # Prepare input files, lists, parameters... for input to 
            # the CL script
            clm=mgr.CLManager(imageIns=ad, suffix="_out", 
                              funcName="mosaicDetectors", log=log)
            
            # Check the status of the CLManager object, 
            # True=continue, False= issue warning
            if not clm.status: 
                raise Errors.ScienceError("One of the inputs has not been " +
                                          "prepared, the " + 
                                          "mosaic_detectors function " +
                                          "can only work on prepared data.")
            
            # Parameters set by the mgr.CLManager or the 
            # definition of the prim 
            clPrimParams = {
                # Retrieve the inputs as a string of filenames
                "inimages"    :clm.imageInsFiles(type="string"),
                "outimages"   :clm.imageOutsFiles(type="string"),
                # Set the value of FL_vardq set above
                "fl_vardq"    :fl_vardq,
                # This returns a unique/temp log file for IRAF 
                "logfile"     :clm.templog.name,
                }
            # Parameters from the Parameter file adjustable by the user
            clSoftcodedParams = {
                # pyrafBoolean converts the python booleans to pyraf ones
                "fl_paste"    :mgr.pyrafBoolean(tile),
                #"outpref"     :suffix,
                "geointer"    :interpolator,
                }
            # Grab the default params dict and update it with 
            # the two above dicts
            clParamsDict = CLDefaultParamsDict("gmosaic")
            clParamsDict.update(clPrimParams)
            clParamsDict.update(clSoftcodedParams)
            
            # Log the parameters that were not defaults
            log.fullinfo("\nParameters set automatically:", 
                         category="parameters")
            # Loop through the parameters in the clPrimParams dictionary
            # and log them
            mgr.logDictParams(clPrimParams)
            
            log.fullinfo("\nParameters adjustable by the user:", 
                         category="parameters")
            # Loop through the parameters in the clSoftcodedParams 
            # dictionary and log them
            mgr.logDictParams(clSoftcodedParams)

            gemini.gmos.gmosaic(**clParamsDict)
            
            if gemini.gmos.gmosaic.status:
                raise Errors.ScienceError("gireduce failed for inputs "+
                             clm.imageInsFiles(type="string"))
            else:
                log.fullinfo("Exited the gmosaic CL script successfully")
            
            # Rename CL outputs and load them back into memory 
            # and clean up the intermediate temp files written to disk
            # refOuts and arrayOuts are None here
            imageOuts, refOuts, arrayOuts = clm.finishCL()
            
            ad_out = imageOuts[0]
            ad_out.filename = ad.filename
            
            # Verify gmosaic was actually run on the file
            # then log file names of successfully reduced files
            if ad_out.phu_get_key_value("GMOSAIC"): 
                log.fullinfo("File "+ad_out.filename+\
                            " was successfully mosaicked")

            # Restore BUNIT, OVERSCAN, AMPNAME keywords
            # to science extension header
            if bunit is not None:
                gt.update_key_value(adinput=ad_out, function="bunit",
                                    value=bunit, extname="SCI")
                if ad_out["VAR"] is not None:
                    gt.update_key_value(adinput=ad_out, function="bunit",
                                        value="%s*%s" % (bunit,bunit),
                                        extname="VAR")
            if avg_overscan is not None:
                for ext in ad_out["SCI"]:
                    ext.set_key_value("OVERSCAN",avg_overscan,
                                      comment="Overscan mean value")
            if all_ampname is not None:
                # These ampnames can be long, so truncate
                # the comment by hand to avoid the error
                # message from pyfits
                comment = "Amplifier name(s)"
                if len(all_ampname)>=65:
                    comment = ""
                else:
                    comment = comment[0:65-len(all_ampname)]
                for ext in ad_out["SCI"]:
                    ext.set_key_value("AMPNAME",all_ampname,
                                      comment=comment)

            # Set DETSEC keyword
            data_shape = ad_out["SCI",1].data.shape
            xbin = ad_out.detector_x_bin()
            if xbin is not None:
                unbin_width = data_shape[1] * xbin
            else:
                unbin_width = data_shape[1]
            if old_detsec is not None:
                new_detsec = "[%i:%i,%i:%i]" % (old_detsec[0]+1,
                                                old_detsec[0]+unbin_width,
                                                old_detsec[2]+1,old_detsec[3])
                ext.set_key_value("DETSEC",new_detsec)
            else:
                ext.set_key_value("DETSEC","")


            # Change type of DQ plane back to int16
            # (gmosaic sets it to float32)
            if ad_out["DQ"] is not None:
                for dqext in ad_out["DQ"]:
                    dqext.data = dqext.data.astype(np.int16)

            # Update GEM-TLM (automatic) and MOSAIC time stamps to the PHU
            # and update logger with updated/added time stamps
            gt.mark_history(adinput=ad_out, keyword=timestamp_key)
            
            adoutput_list.append(ad_out)
        
        # Return the outputs list, even if there is only one output
        return adoutput_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise


def tile_arrays(adinput=None, tile_all=False):

    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()

    # The validate_input function ensures that adinput is not None and returns
    # a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)

    # Define the keyword to be used for the time stamp for this user level
    # function
    timestamp_key = timestamp_keys["tile_arrays"]

    # Initialize the list of output AstroData objects
    adoutput_list = []

    try:
        # Loop over each input AstroData object in the input list
        for ad in adinput:

            # Store PHU to pass to output AD
            phu = ad.phu

            nsciext = ad.count_exts("SCI")
            if nsciext==1:
                log.fullinfo("Only one science extension found; " +
                             "no tiling done for %s" % ad.filename)
                adoutput_list.append(ad)
            else:

                # First trim off any overscan regions still present
                # so they won't get tiled with science data
                if not ad.phu_get_key_value(timestamp_keys["trim_overscan"]):
                    ad = bs.trim_overscan(adinput=ad)[0]

                # Make chip gaps to tile with science extensions if tiling all
                # Gap width should come from lookup table
                gap_height = int(ad["SCI",1].data.shape[0])
                gap_width = _obtain_arraygap(adinput=ad)
                chip_gap = np.zeros((gap_height,gap_width))


                # Get the correct order of the extensions by sorting on
                # first element in detector section
                # (raw ordering is whichever amps read out first)
                detsecs = ad.detector_section().as_list()
                if not isinstance(detsecs[0],list):
                    detsecs = [detsecs]
                detx1 = [sec[0] for sec in detsecs]
                ampsorder = range(1,nsciext+1)
                orderarray = np.array(zip(ampsorder,
                                          detx1),dtype=[('ext',np.int),
                                                        ('detx1',np.int)])
                orderarray.sort(order='detx1')
                if np.all(ampsorder==orderarray['ext']):
                    in_order = True
                else:
                    ampsorder = orderarray['ext']
                    in_order = False

                # Get array sections for determining when
                # a new array is found
                ccdsecs = ad.array_section().as_list()
                if not isinstance(ccdsecs[0],list):
                    ccdsecs = [ccdsecs]
                if len(ccdsecs)!=nsciext:
                    ccdsecs*=nsciext
                ccdx1 = [sec[0] for sec in ccdsecs]


                # Now, get the number of extensions per ccd

                # Initialize everything 
                ccd_data = {}
                amps_per_ccd = {}
                sci_data_list = []
                var_data_list = []
                dq_data_list = []
                num_ccd = 0
                ext_count = 1
                ampname = {}
                amplist = []
                refsec = {}
                mapping_dict = {}
                
                # Initialize these so that first extension will always
                # start a new CCD
                last_detx1 = detx1[ampsorder[0]-1]-1
                last_ccdx1 = ccdx1[ampsorder[0]-1]

                for i in ampsorder:
                    sciext = ad["SCI",i]
                    varext = ad["VAR",i]
                    dqext = ad["DQ",i]

                    this_detx1 = detx1[i-1]
                    this_ccdx1 = ccdx1[i-1]

                    amp = sciext.get_key_value("AMPNAME")

                    if (this_detx1>last_detx1 and this_ccdx1<=last_ccdx1):
                        # New CCD found

                        # If not first extension, store current data lists
                        # (or, if tiling all CCDs together, add a chip gap)
                        if num_ccd>0:
                            if tile_all:
                                sci_data_list.append(chip_gap)
                                if varext is not None:
                                    var_data_list.append(chip_gap)
                                if dqext is not None:
                                    dq_data_list.append(chip_gap)
                            else:
                                ccd_data[num_ccd] = {"SCI":sci_data_list,
                                                     "VAR":var_data_list,
                                                     "DQ":dq_data_list}
                                ampname[num_ccd] = amplist

                        # Increment CCD number and restart amps per ccd
                        num_ccd += 1
                        amps_per_ccd[num_ccd] = 1

                        # Start new data lists (or append if tiling all)
                        if tile_all:                            
                            sci_data_list.append(sciext.data)
                            if varext is not None:
                                var_data_list.append(varext.data)
                            if dqext is not None:
                                dq_data_list.append(dqext.data)

                            # Keep the name of the amplifier
                            # (for later header updates)
                            amplist.append(amp)

                            # Keep ccdsec and detsec from first extension only
                            if num_ccd==1:
                                refsec[1] = {"CCD":ccdsecs[i-1],
                                             "DET":detsecs[i-1]} 
                        else:
                            sci_data_list = [sciext.data]
                            if varext is not None:
                                var_data_list = [varext.data]
                            if dqext is not None:
                                dq_data_list = [dqext.data]
                            amplist = [amp]
                            # Keep ccdsec and detsec from first extension
                            # of each CCD
                            refsec[num_ccd] = {"CCD":ccdsecs[i-1],
                                               "DET":detsecs[i-1]}
                    else:
                        # Increment amps and append data
                        amps_per_ccd[num_ccd] += 1
                        amplist.append(amp)
                        sci_data_list.append(sciext.data)
                        if varext:
                            var_data_list.append(varext.data)
                        if dqext:
                            dq_data_list.append(dqext.data)
                        

                    # If last iteration, store the current data lists
                    if tile_all:
                        key = 1
                    else:
                        key = num_ccd
                    if ext_count==nsciext:
                        ccd_data[key] = {"SCI":sci_data_list,
                                         "VAR":var_data_list,
                                         "DQ":dq_data_list}
                        ampname[key] = amplist

                    # Keep track of which extensions ended up in
                    # which CCD
                    try:
                        mapping_dict[key].append(i)
                    except KeyError:
                        mapping_dict[key] = [i]

                    last_ccdx1 = this_ccdx1
                    last_detx1 = this_detx1
                    ext_count += 1

                if nsciext==num_ccd and in_order and not tile_all:
                    # No reordering or tiling necessary, return input AD
                    adoutput = ad
                    log.fullinfo("Only one amplifier per array; " +
                                 "no tiling done for %s" % ad.filename)

                else:
                    if not in_order:
                        log.fullinfo("Reordering data by detector section")
                    if tile_all:
                        log.fullinfo("Tiling all data into one extension")
                    elif nsciext!=num_ccd:
                        log.fullinfo("Tiling data into one extension per array")

                    # Get header from the center-left extension of each CCD
                    # (or the center of CCD2 if tiling all)
                    # This is in order to get the most accurate WCS on CCD2
                    ref_header = {}
                    startextn = 1
                    ref_shift = {}
                    ref_shift_temp = 0
                    total_shift = 0
                    on_ext=0
                    for ccd in range(1,num_ccd+1):
                        if tile_all:
                            key = 1
                            if ccd!=2:
                                startextn += amps_per_ccd[ccd]
                                continue
                        else:
                            key = ccd
                            total_shift = 0

                        refextn = ampsorder[int((amps_per_ccd[ccd]+1)/2.0-1)
                                            + startextn - 1]

                        # Get size of reference shift from 0,0 to
                        # start of reference extension
                        for data in ccd_data[key]["SCI"]:
                            # if it's a chip gap, add width to total, continue
                            if data.shape[1]==gap_width:
                                total_shift += gap_width
                            else:
                                on_ext+=1
                                # keep total up to now if it's the reference ext
                                if ampsorder[on_ext-1]==refextn:
                                    ref_shift_temp = total_shift
                                # add in width of this extension
                                total_shift += data.shape[1]

                        # Get header from reference extension
                        dict = {}
                        for extname in ["SCI","VAR","DQ"]:
                            ext = ad[extname,refextn]
                            if ext is not None:
                                header = ext.header
                            else:
                                header = None
                            dict[extname] = header
                        
                        if dict["SCI"] is None:
                            raise Errors.ScienceError("Header not found " +
                                                      "for reference " + 
                                                      "extension " +
                                                      "[SCI,%i]" % refextn)

                        ref_header[key] = dict
                        ref_shift[key] = ref_shift_temp

                        startextn += amps_per_ccd[ccd]

                    # Make a new AD
                    adoutput = AstroData()
                    adoutput.filename = ad.filename
                    adoutput.phu = phu

                    # Stack data from each array together and
                    # append to output AD
                    if tile_all:
                        num_ccd = 1
                    nextend = 0
                    for ccd in range(1,num_ccd+1):
                        for extname in ccd_data[ccd].keys():
                            if len(ccd_data[ccd][extname])>0:
                                data = np.hstack(ccd_data[ccd][extname])
                                header = ref_header[ccd][extname]
                                new_ext = AstroData(data=data,header=header)
                                new_ext.rename_ext(name=extname,ver=ccd)
                                adoutput.append(new_ext)
                            
                                nextend += 1

                    # Update header keywords with appropriate values
                    # for the new data set
                    adoutput.phu_set_key_value("NSCIEXT",num_ccd)
                    adoutput.phu_set_key_value("NEXTEND",nextend)
                    for ext in adoutput:
                        extname = ext.extname()
                        extver = ext.extver()

                        # Update AMPNAME
                        if extname!="DQ":
                            new_ampname = ",".join(ampname[extver])

                            # These ampnames can be long, so truncate
                            # the comment by hand to avoid the error
                            # message from pyfits
                            comment = "Amplifier name(s)"
                            if len(new_ampname)>=65:
                                comment = ""
                            else:
                                comment = comment[0:65-len(new_ampname)]
                            ext.set_key_value("AMPNAME",new_ampname,
                                              comment=comment)
                        # Update DATASEC
                        data_shape = ext.data.shape
                        new_datasec = "[1:%i,1:%i]" % (data_shape[1],
                                                       data_shape[0])
                        ext.set_key_value("DATASEC",new_datasec)

                        # Update DETSEC
                        unbin_width = data_shape[1] * ad.detector_x_bin()
                        old_detsec = refsec[extver]["DET"]
                        new_detsec = "[%i:%i,%i:%i]" % (old_detsec[0]+1,
                                                  old_detsec[0]+unbin_width,
                                                  old_detsec[2]+1,old_detsec[3])
                        ext.set_key_value("DETSEC",new_detsec)

                        # Update CCDSEC
                        old_ccdsec = refsec[extver]["CCD"]
                        new_ccdsec = "[%i:%i,%i:%i]" % (old_ccdsec[0]+1,
                                                  old_ccdsec[0]+unbin_width,
                                                  old_ccdsec[2]+1,old_ccdsec[3])
                        ext.set_key_value("CCDSEC",new_ccdsec)

                        # Update CRPIX1
                        crpix1 = ext.get_key_value("CRPIX1")
                        if crpix1 is not None:
                            new_crpix1 = crpix1 + ref_shift[extver]
                            ext.set_key_value("CRPIX1",new_crpix1)

                    
                    # Update and attach OBJCAT if needed
                    if ad["OBJCAT"] is not None:
                        adoutput = _tile_objcat(ad,adoutput,mapping_dict)[0]

                    # Refresh AstroData types in output file (original ones
                    # were lost when new AD was created)
                    adoutput.refresh_types()

                # Add the appropriate time stamps to the PHU
                gt.mark_history(adinput=adoutput, keyword=timestamp_key)

                # Append the output AstroData object to the list of output
                # AstroData objects
                adoutput_list.append(adoutput)
            
        # Return the list of output AstroData objects
        return adoutput_list

    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise


##############################################################################
# Below are the helper functions for the user level functions in this module #
##############################################################################

def _obtain_arraygap(adinput=None):
    """
    This function obtains the raw array gap size for the different GMOS
    detectors and returns it after correcting for binning. There are two
    values in the GMOSArrayGaps.py file in the GMOS
    lookup directory, one for unbinned data and one to be used to calculate
    the chip gap when the data are binned.
    """
    
    # Get the dictionary containing the CCD gaps
    all_arraygaps_dict = Lookups.get_lookup_table(\
        "Gemini/GMOS/GMOSArrayGaps.py","gmosArrayGaps")
    
    # Obtain the X binning and detector type for the ad input
    detector_x_bin = adinput.detector_x_bin()
    detector_type = adinput.phu_get_key_value("DETTYPE")

    # Check the read values
    if detector_x_bin is None or detector_type is None:
        if hasattr(ad, "exception_info"):
            raise adinput.exception_info
    
    # Check if the data are binned
    if detector_x_bin > 1:
        bin_string = "binned"
    else:
        bin_string = "unbinned"

    # Form the key
    key = (detector_type, bin_string)

    # Obtain the array gap value and fix for any binning
    if key in all_arraygaps_dict:
        arraygap = all_arraygaps_dict[key] / detector_x_bin.as_pytype()
    else:
        raise Errors.ScienceError("Array gap value not " +
                              "found for %s" % (detector_type)) 
    return arraygap

def _tile_objcat(adinput=None,adoutput=None,mapping_dict=None):
    """
    This function tiles together separate OBJCAT extensions, converting
    the pixel coordinates to the new WCS.
    """

    from gempy.science import photometry as ph

    adinput = gt.validate_input(adinput=adinput)
    adoutput = gt.validate_input(adinput=adoutput)

    if mapping_dict is None:
        raise Errors.InputError("mapping_dict must not be None")

    if len(adinput)!=len(adoutput):
        raise Errors.InputError("adinput must have same length as adoutput")
    output_dict = gt.make_dict(key_list=adinput, value_list=adoutput)

    adoutput_list = []
    for ad in adinput:
        
        adout = output_dict[ad]

        objcat = ad["OBJCAT"]
        if objcat is None:
            raise Errors.InputError("No OBJCAT found in %s" % ad.filename)

        for outext in adout["SCI"]:
            out_extver = outext.extver()
            output_wcs = pywcs.WCS(outext.header)

            col_names = None
            col_fmts = None
            col_data = {}
            for inp_extver in mapping_dict[out_extver]:
                inp_objcat = ad["OBJCAT",inp_extver]

                # Make sure there is data in the OBJCAT
                if inp_objcat is None:
                    continue
                if inp_objcat.data is None:
                    continue
                if len(inp_objcat.data)==0:
                    continue

                # Get column names, formats from first OBJCAT
                if col_names is None:
                    col_names = inp_objcat.data.names
                    col_fmts = inp_objcat.data.formats
                    for name in col_names:
                        col_data[name] = inp_objcat.data.field(name).tolist()
                else:
                    # Stack all OBJCAT data together
                    for name in col_names:
                        col_data[name].extend(inp_objcat.data.field(name))

            # Get new pixel coordinates for the objects from RA/Dec
            # and the output WCS
            ra = col_data["X_WORLD"]
            dec = col_data["Y_WORLD"]
            newx,newy = output_wcs.wcs_sky2pix(ra,dec,1)
            col_data["X_IMAGE"] = newx
            col_data["Y_IMAGE"] = newy

            columns = {}
            for name,format in zip(col_names,col_fmts):
                # Let add_objcat auto-number sources
                if name=="NUMBER":
                    continue

                # Define pyfits column to pass to add_objcat
                columns[name] = pf.Column(name=name,format=format,
                                          array=col_data[name])


            adout = ph.add_objcat(adinput=adout, extver=out_extver,
                                  columns=columns)[0]

        adoutput_list.append(adout)

    return adoutput_list
