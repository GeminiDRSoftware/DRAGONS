import numpy as np
import pywcs
from astrodata.utils import Errors
from astrodata.utils import logutils
from gempy.library import astrotools as at
from gempy.gemini import gemini_tools as gt
from primitives_GENERAL import GENERALPrimitives

class ResamplePrimitives(GENERALPrimitives):
    """
    This is the class containing all of the primitives for the GEMINI level of
    the type hierarchy tree. It inherits all the primitives from the level
    above, 'GENERALPrimitives'.
    """
    astrotype = "GEMINI"
    
    def init(self, rc):
        GENERALPrimitives.init(self, rc)
        return rc
    init.pt_hide = True
    
    def alignToReferenceFrame(self, rc):
        """
        This primitive applies the transformation encoded in the input images
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
        
        :param interpolator: type of interpolation desired
        :type interpolator: string, possible values are None, 'nearest', 
                            'linear', 'spline2', 'spline3', 'spline4', 
                            or 'spline5'
        
        :param trim_data: flag to indicate whether output image should be trimmed
                          to the size of the reference image.
        :type trim_data: Boolean
        """
        
        # Instantiate the log
        log = logutils.get_logger(__name__)

        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "alignToReferenceFrame",
                                 "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["alignToReferenceFrame"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Check whether two or more input AstroData objects were provided
        adinput = rc.get_inputs_as_astrodata()
        if len(adinput) <= 1:
            log.warning("No alignment will be performed, since at least two " \
                        "input AstroData objects are required for " \
                        "alignToReferenceFrame")
            # Set the input AstroData object list equal to the output AstroData
            # objects list without further processing
            adoutput_list = adinput
        else:
            
            # Get the necessary parameters from the RC
            interpolator = rc["interpolator"]
            trim_data = rc["trim_data"]

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
            
            # get reference WCS and shape
            reference = adinput[0]
            ref_wcs = pywcs.WCS(reference["SCI"].header)
            ref_shape = reference["SCI"].data.shape
            ref_corners = at.get_corners(ref_shape)
            naxis = len(ref_shape)

            # first pass: get output image shape required to fit all
            # data in output by transforming corner coordinates of images
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
                    x2y2 = img_wcs.wcs_sky2pix(
                        ref_wcs.wcs_pix2sky([x1y1],1),1)[0]
                    
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
                    
                    img_corners = [(corner[1],corner[0]) 
                                   for corner in xy_corners]
                
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
                    cvals = [
                        corner[axis] for ic in all_corners for corner in ic]
                    out_shape.append(int(max(cvals)-min(cvals)+1))
                    
                    # if just shifting, need to set centering shift
                    # for reference image from offsets already calculated
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

            # shift the reference image to keep it in the center
            # of the new array (do the same for VAR and DQ)

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
                
                # Make a blank data array to transform into
                if ext.extname()=="DQ":
                    # pad the DQ plane with 1 instead of 0, and make the data
                    # type int16
                    trans_data = np.zeros(out_shape).astype(np.int16)
                    trans_data += 1
                else:
                    trans_data = np.zeros(out_shape).astype(np.float32)
                
                trans_data[int(-cenoff[0]):int(ref_shape[0]-cenoff[0]),
                           int(-cenoff[1]):int(ref_shape[1]-cenoff[1])] = \
                           ref_data
                
                ext.data = trans_data
                
                # update the WCS in the reference image to account for the shift
                ext.set_key_value("CRPIX1", ref_wcs.wcs.crpix[0]-cenoff[1],
                                  comment=self.keyword_comments["CRPIX1"])
                ext.set_key_value("CRPIX2", ref_wcs.wcs.crpix[1]-cenoff[0],
                                  comment=self.keyword_comments["CRPIX2"])
                
                # set area keywords
                for key in area_keys:
                    ext.set_key_value(key[0],key[1],key[2])
            
            # update the WCS in the PHU as well
            reference.phu_set_key_value(
                "CRPIX1", ref_wcs.wcs.crpix[0]-cenoff[1],
                comment=self.keyword_comments["CRPIX1"])
            reference.phu_set_key_value(
                "CRPIX2", ref_wcs.wcs.crpix[1]-cenoff[0],
                comment=self.keyword_comments["CRPIX2"])
            
            out_wcs = pywcs.WCS(reference["SCI"].header)
            
            # Change the reference filename and append it to the output list
            reference.filename = gt.filename_updater(
                adinput=reference, suffix=rc["suffix"], strip=True)
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
                    x2y2 = img_wcs.wcs_sky2pix(
                        out_wcs.wcs_pix2sky([x1y1],1),1)[0]
                    
                    shift = np.roll(np.rint(x2y2-x1y1),1)
                    if np.any(shift>0):
                        log.warning("Shift was calculated to be >0; " +
                                    "interpolator=None "+
                                    "may not be appropriate for this data.")
                        shift = np.where(shift>0,0,shift)
                    
                    # update PHU WCS keywords
                    log.fullinfo("Offsets: "+repr(np.roll(shift,1)))
                    log.fullinfo("Updating WCS to track shift in data")
                    ad.phu_set_key_value(
                        "CRPIX1", img_wcs.wcs.crpix[0]-shift[1],
                        comment=self.keyword_comments["CRPIX1"])
                    ad.phu_set_key_value(
                        "CRPIX2", img_wcs.wcs.crpix[1]-shift[0],
                        comment=self.keyword_comments["CRPIX2"])
                
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
                    # reference pixel and adding the input reference pixel
                    # back in
                    refcrpix = np.roll(out_wcs.wcs.crpix,1)
                    imgcrpix = np.roll(img_wcs.wcs.crpix,1)
                    offset = imgcrpix - np.dot(matrix,refcrpix)
                    
                    # then add in the shift of origin due to dithering offset.
                    # This is the transform of the reference CRPIX position,
                    # minus the original position
                    trans_crpix = img_wcs.wcs_sky2pix(
                        out_wcs.wcs_pix2sky([out_wcs.wcs.crpix],1),1)[0]
                    trans_crpix = np.roll(trans_crpix,1)
                    offset = offset + trans_crpix-imgcrpix
                    
                    # Since the transformation really is into the reference
                    # WCS coordinate system as near as possible, just set image
                    # WCS equal to reference WCS
                    log.fullinfo("Offsets: "+repr(np.roll(offset,1)))
                    log.fullinfo("Transformation matrix:\n"+repr(matrix))
                    log.fullinfo("Updating WCS to match reference WCS")
                    ad.phu_set_key_value(
                        "CRPIX1", out_wcs.wcs.crpix[0],
                        comment=self.keyword_comments["CRPIX1"])
                    ad.phu_set_key_value(
                        "CRPIX2", out_wcs.wcs.crpix[1],
                        comment=self.keyword_comments["CRPIX2"])
                    ad.phu_set_key_value(
                        "CRVAL1", out_wcs.wcs.crval[0],
                        comment=self.keyword_comments["CRVAL1"])
                    ad.phu_set_key_value(
                        "CRVAL2", out_wcs.wcs.crval[1],
                        comment=self.keyword_comments["CRVAL2"])
                    ad.phu_set_key_value(
                        "CD1_1", out_wcs.wcs.cd[0,0],
                        comment=self.keyword_comments["CD1_1"])
                    ad.phu_set_key_value(
                        "CD1_2", out_wcs.wcs.cd[0,1],
                        comment=self.keyword_comments["CD1_2"])
                    ad.phu_set_key_value(
                        "CD2_1", out_wcs.wcs.cd[1,0],
                        comment=self.keyword_comments["CD2_1"])
                    ad.phu_set_key_value(
                        "CD2_2", out_wcs.wcs.cd[1,1],
                        comment=self.keyword_comments["CD2_2"])
                
                # transform corners to find new location of original data
                data_corners = out_wcs.wcs_sky2pix(
                    img_wcs.wcs_pix2sky(xy_img_corners[i-1],0),1)
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
                        
                        # Make a blank data array to transform into
                        if extname=="DQ":
                            # pad the DQ plane with 1 instead of 0, and
                            # make the data type int16
                            trans_data = np.zeros(out_shape).astype(np.int16)
                            trans_data += 1
                        else:
                            trans_data = np.zeros(out_shape).astype(np.float32)
                        
                        trans_data[int(-shift[0]):int(img_shape[0]
                                                      -shift[0]),
                                   int(-shift[1]):int(img_shape[1]
                                                      -shift[1])] = img_data
                        
                        matrix_det = 1.0
                        
                        # update the wcs to track the transformation
                        ext.set_key_value(
                            "CRPIX1", img_wcs.wcs.crpix[0]-shift[1],
                            comment=self.keyword_comments["CRPIX1"])
                        ext.set_key_value(
                            "CRPIX2", img_wcs.wcs.crpix[1]-shift[0],
                            comment=self.keyword_comments["CRPIX2"])
                    
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
                            unpack_data = np.unpackbits(
                                np.uint8(img_data)).reshape(unp)
                            
                            # transform each mask
                            trans_data = np.zeros(out_shape).astype(np.int16)
                            for j in range(0,8):
                                
                                # skip the transformation if there are no flags
                                # set (but always do the bad pixel mask because
                                # it is needed to mask the part of the array
                                #  that was padded out to match the reference
                                # image)
                                if not unpack_data[:,:,j].any() and j!=7:
                                    # first bit is j=7 because unpack
                                    # is backwards 
                                    continue
                                
                                mask = np.float32(unpack_data[:,:,j])
                                
                                # if bad pix bit, pad with 1. 
                                # Otherwise, pad with 0
                                if j==7:
                                    cval = 1
                                else:
                                    cval = 0
                                trans_mask = affine_transform(
                                    mask, matrix, offset=offset,
                                    output_shape=out_shape, order=order,
                                    cval=cval)
                                del mask; mask = None
                                
                                # flag any pixels with >1% influence
                                # from bad pixel
                                trans_mask = np.where(np.abs(trans_mask)>0.01,
                                                      2**(7-j),0)
                                
                                # add the flags into the overall mask
                                trans_data += trans_mask
                                del trans_mask; trans_mask = None
    
                        else:
                            
                            # transform science and variance data in the
                            # same way
                            cval = 0.0
                            trans_data = affine_transform(
                                img_data, matrix, offset=offset,
                                output_shape=out_shape, order=order, cval=cval)
                        
                        # update the wcs
                        ext.set_key_value(
                            "CRPIX1", out_wcs.wcs.crpix[0],
                            comment=self.keyword_comments["CRPIX1"])
                        ext.set_key_value(
                            "CRPIX2", out_wcs.wcs.crpix[1],
                            comment=self.keyword_comments["CRPIX2"])
                        ext.set_key_value(
                            "CRVAL1", out_wcs.wcs.crval[0],
                            comment=self.keyword_comments["CRVAL1"])
                        ext.set_key_value(
                            "CRVAL2", out_wcs.wcs.crval[1],
                            comment=self.keyword_comments["CRVAL2"])
                        ext.set_key_value(
                            "CD1_1", out_wcs.wcs.cd[0,0],
                            comment=self.keyword_comments["CD1_1"])
                        ext.set_key_value(
                            "CD1_2", out_wcs.wcs.cd[0,1],
                            comment=self.keyword_comments["CD1_2"])
                        ext.set_key_value(
                            "CD2_1", out_wcs.wcs.cd[1,0],
                            comment=self.keyword_comments["CD2_1"])
                        ext.set_key_value(
                            "CD2_2", out_wcs.wcs.cd[1,1],
                            comment=self.keyword_comments["CD2_2"])
                        
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
                gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)

                # Change the filename
                ad.filename = gt.filename_updater(adinput=ad, 
                                                  suffix=rc["suffix"], 
                                                  strip=True)

                # Append the output AstroData object to the list
                # of output AstroData objects
                adoutput_list.append(ad)
        
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
