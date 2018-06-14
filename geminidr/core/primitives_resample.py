#
#                                                                  gemini_python
#
#                                                         primitives_resample.py
# ------------------------------------------------------------------------------
import numpy as np
from astropy.wcs import WCS
from scipy.ndimage import affine_transform

from gempy.library import astrotools as at
from gempy.gemini import gemini_tools as gt
from gempy.utils import logutils

from geminidr.gemini.lookups import DQ_definitions as DQ

from geminidr import PrimitivesBASE
from . import parameters_resample

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
interpolators = {"nearest": 0,
                 "linear": 1,
                 "spline2": 2,
                 "spline3": 3,
                 "spline4": 4,
                 "spline5": 5,
                 }
# ------------------------------------------------------------------------------
@parameter_override
class Resample(PrimitivesBASE):
    """
    This is the class containing all of the primitives for resampling.
    """
    tagset = None

    def __init__(self, adinputs, **kwargs):
        super(Resample, self).__init__(adinputs, **kwargs)
        self._param_update(parameters_resample)

    def resampleToCommonFrame(self, adinputs=None, **params):
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
        plane is padded with 16s.
        
        The WCS keywords in the headers of the output images are updated
        to reflect the transformation.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        interpolator: str
            desired interpolation [nearest | linear | spline2 | spline3 |
                                   spline4 | spline5]
        trim_data: bool
            trim image to size of reference image?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        interpolator = params["interpolator"]
        trim_data = params["trim_data"]
        sfx = params["suffix"]

        if len(adinputs) < 2:
            log.warning("No alignment will be performed, since at least two "
                        "input AstroData objects are required for "
                        "resampleToCommonFrame")
            return adinputs

        if not all(len(ad)==1 for ad in adinputs):
            raise IOError("All input images must have only one extension.")

        # --------------------  BEGIN establish reference frame  -------------------
        ref_image = adinputs[0]
        ref_wcs = WCS(ref_image[0].hdr)
        ref_shape = ref_image[0].data.shape
        ref_corners = at.get_corners(ref_shape)
        naxis = len(ref_shape)

        # first pass: get output image shape required to fit all
        # data in output by transforming corner coordinates of images
        all_corners = [ref_corners]
        corner_values = _transform_corners(adinputs[1:], all_corners, ref_wcs,
                                           interpolator)
        all_corners, xy_img_corners, shifts = corner_values
        refoff, out_shape = _shifts_and_shapes(all_corners, ref_shape, naxis,
                                               interpolator, trim_data, shifts)
        ref_corners = [(corner[1] - refoff[1] + 1, corner[0] - refoff[0] + 1) # x,y
                       for corner in ref_corners]
        area_keys = _build_area_keys(ref_corners)

        ref_image.hdr.set('CRPIX1', ref_wcs.wcs.crpix[0]-refoff[1],
                          self.keyword_comments["CRPIX1"])
        ref_image.hdr.set('CRPIX2', ref_wcs.wcs.crpix[1]-refoff[0],
                          self.keyword_comments["CRPIX2"])
        padding = tuple((int(-cen),out-int(ref-cen)) for cen, out, ref in
                        zip(refoff, out_shape, ref_shape))
        _pad_image(ref_image, padding)

        for key in area_keys:
            ref_image[0].hdr.set(*key)
        out_wcs = WCS(ref_image[0].hdr)
        ref_image.update_filename(suffix=sfx, strip=True)
        # -------------------- END establish reference frame -----------------------

        # --------------------   BEGIN transform data ...  -------------------------
        for ad, corners in zip(adinputs[1:], xy_img_corners):
            if interpolator:
                trans_parameters = _composite_transformation_matrix(ad,
                                        out_wcs, self.keyword_comments)
                matrix, matrix_det, img_wcs, offset = trans_parameters
            else:
                shift = _composite_from_ref_wcs(ad, out_wcs,
                                                self.keyword_comments)
                matrix_det = 1.0

            # transform corners to find new location of original data
            data_corners = out_wcs.all_world2pix(
                img_wcs.all_pix2world(corners, 0), 1)
            area_keys = _build_area_keys(data_corners)

            if interpolator:
                kwargs = {'matrix': matrix, 'offset': offset,
                          'order': interpolators[interpolator],
                          'output_shape': out_shape}
                new_var = None if ad[0].variance is None else \
                    affine_transform(ad[0].variance, cval=0.0, **kwargs)
                new_mask = None if ad[0].mask is None else \
                    _transform_mask(ad[0].mask, **kwargs)
                if hasattr(ad[0], 'OBJMASK'):
                    ad[0].OBJMASK = _transform_mask(ad[0].OBJMASK, **kwargs)
                ad[0].reset(affine_transform(ad[0].data, cval=0.0, **kwargs),
                            new_mask, new_var)
            else:
                padding = tuple((int(-s), out-int(img-s)) for s, out, img in
                                zip(shift, out_shape, ad[0].data.shape))
                _pad_image(ad, padding)

            if abs(1.0 - matrix_det) > 1e-6:
                    log.fullinfo("Multiplying by {} to conserve flux".format(matrix_det))
                    # Allow the arith toolbox to do the multiplication
                    # so that variance is handled correctly
                    ad.multiply(matrix_det)

            for key in area_keys:
                ref_image[0].hdr.set(*key)

            # Timestamp and update filename
            ad.update_filename(suffix=sfx, strip=True)
        return adinputs

# =================================== prive ====================================
def _transform_corners(ads, all_corners, ref_wcs, interpolator):
    shifts = []
    xy_img_corners = []

    for ad in ads:
        img_wcs = WCS(ad[0].hdr)
        img_shape = ad[0].data.shape
        img_corners = at.get_corners(img_shape)
        xy_corners  = [(corner[1],corner[0]) for corner in img_corners]
        xy_img_corners.append(xy_corners)

        if interpolator is None:
            # find shift by transforming center position of field
            # (so that center matches best)
            x1y1 = np.array([img_shape[1]/2.0, img_shape[0]/2.0])
            x2y2 = img_wcs.all_world2pix(ref_wcs.all_pix2world([x1y1],1), 1)[0]

            # round shift to nearest integer and flip x and y
            offset = np.roll(np.rint(x2y2-x1y1),1)

            # shift corners of image
            img_corners = [tuple(offset+corner) for corner in img_corners]
            shifts.append(offset)
        else:
            # transform corners of image via WCS
            xy_corners = img_wcs.all_world2pix(ref_wcs.all_pix2world(xy_corners,0),0)
            img_corners = [(corner[1],corner[0]) for corner in xy_corners]

        all_corners.append(img_corners)
    return all_corners, xy_img_corners, shifts

def _shifts_and_shapes(all_corners, ref_shape, naxis, interpolator, trim_data, shifts):
    """
    all_corners are locations (y,x) of all 4 corners of the reference image in the
    pixel space of each image
    """
    log = logutils.get_logger(__name__)
    if trim_data:
        refoff=[0]*naxis
        out_shape = ref_shape
        log.fullinfo("Trimming data to size of reference image")
    else:
        log.fullinfo("Growing reference image to keep all data; "
                     "centering data, and updating WCS to account "
                     "for shift")

        # Otherwise, use the corners of the images to get the minimum
        # required output shape to hold all data
        out_shape = []
        refoff = []
        for axis in range(naxis):
            # get output shape from corner values
            cvals = [corner[axis] for ic in all_corners for corner in ic]
            out_shape.append(int(max(cvals)-min(cvals)+1))
            refoff.append(-max(0, int(max(cvals) - ref_shape[axis] + 1)))

            # if just shifting, need to set centering shift
            # for reference image from offsets already calculated
            if interpolator is None:
                svals = [shift[axis] for shift in shifts]
                # include a 0 shift for the reference image
                # (in case it's already centered)
                svals.append(0.0)
                refoff.append(-int(max(svals)))

        out_shape = tuple(out_shape)
        log.fullinfo("New output shape: "+repr(out_shape))

        # if not shifting, get offset required to center reference image
        # from the size of the image
        #if interpolator:
        #    incen = [0.5*(axlen-1) for axlen in ref_shape]
        #    outcen = [0.5*(axlen-1) for axlen in out_shape]
        #    cenoff = np.rint(incen) - np.rint(outcen)
    return refoff, out_shape

def _build_area_keys(corners):
    log = logutils.get_logger(__name__)
    log.fullinfo("Setting AREA keywords to denote original data area.")
    log.fullinfo("AREATYPE = 'P4'     / Polygon with 4 vertices")
    area_keys = [("AREATYPE", "P4", "Polygon with 4 vertices")]
    for i in range(len(corners)):
        for axis in range(len(corners[i])):
            key_name = "AREA{}_{}".format(i+1, axis+1)
            key_value = corners[i][axis]
            key_comment = "Vertex {}, dimension {}".format(i+1, axis+1)
            area_keys.append((key_name, key_value, key_comment))
            log.fullinfo("{:8s} = {:7.2f}  / {}".format(key_name, key_value,
                                                        key_comment))
    return area_keys

def _composite_transformation_matrix(ad, out_wcs, keyword_comments):
    log = logutils.get_logger(__name__)
    img_wcs = WCS(ad[0].hdr)
    # get transformation matrix from composite of wcs's
    # matrix = in_sky2pix*out_pix2sky (converts output to input)
    xy_matrix = np.dot(np.linalg.inv(img_wcs.wcs.cd), out_wcs.wcs.cd)
    # switch x and y for compatibility with numpy ordering
    flip_xy = np.roll(np.eye(2), 2)
    matrix = np.dot(flip_xy,np.dot(xy_matrix, flip_xy))
    matrix_det = np.linalg.det(matrix)

    # offsets: shift origin of transformation to the reference
    # pixel by subtracting the transformation of the output
    # reference pixel and adding the input reference pixel
    # back in
    refcrpix = np.roll(out_wcs.wcs.crpix, 1)
    imgcrpix = np.roll(img_wcs.wcs.crpix, 1)
    offset = imgcrpix - np.dot(matrix, refcrpix)

    # then add in the shift of origin due to dithering offset.
    # This is the transform of the reference CRPIX position,
    # minus the original position
    trans_crpix = img_wcs.all_world2pix(
                  out_wcs.all_pix2world([out_wcs.wcs.crpix],1), 1)[0]
    trans_crpix = np.roll(trans_crpix, 1)
    offset = offset + trans_crpix-imgcrpix

    # Since the transformation really is into the reference
    # WCS coordinate system as near as possible, just set image
    # WCS equal to reference WCS
    log.fullinfo("Offsets: "+repr(np.roll(offset, 1)))
    log.fullinfo("Transformation matrix:\n"+repr(matrix))
    log.fullinfo("Updating WCS to match reference WCS")

    for ax in (1, 2):
        ad.hdr.set('CRPIX{}'.format(ax), out_wcs.wcs.crpix[ax-1],
                   comment=keyword_comments["CRPIX{}".format(ax)])
        ad.hdr.set('CRVAL{}'.format(ax), out_wcs.wcs.crval[ax-1],
                    comment=keyword_comments["CRVAL{}".format(ax)])
        for ax2 in (1, 2):
            ad.hdr.set('CD{}_{}'.format(ax,ax2), out_wcs.wcs.cd[ax-1,ax2-1],
                       comment=keyword_comments["CD{}_{}".format(ax,ax2)])

    return (matrix, matrix_det, img_wcs, offset) # ad ?

def _composite_from_ref_wcs(ad, out_wcs, keyword_comments):
    log = logutils.get_logger(__name__)
    img_wcs = WCS(ad[0].hdr)
    img_shape = ad[0].data.shape

    # recalculate shift from new reference wcs
    x1y1 = np.array([img_shape[1] / 2.0, img_shape[0]/2.0])
    x2y2 = img_wcs.all_world2pix(out_wcs.all_pix2world([x1y1], 1), 1)[0]
    shift = np.roll(np.rint(x2y2 - x1y1), 1)
    if np.any(shift > 0):
        log.warning("Shift was calculated to be > 0; interpolator=None "
                    "may not be appropriate for this data.")
        shift = np.where(shift > 0, 0, shift)

    # update PHU WCS keywords
    log.fullinfo("Offsets: " + repr(np.roll(shift, 1)))
    log.fullinfo("Updating WCS to track shift in data")
    ad.hdr.set("CRPIX1", img_wcs.wcs.crpix[0] - shift[1],
                         comment=keyword_comments["CRPIX1"])
    ad.hdr.set("CRPIX2", img_wcs.wcs.crpix[1] - shift[0],
                         comment=keyword_comments["CRPIX2"])
    return shift  # ad ?

def _pad_image(ad, padding):
    """Pads the image with zeroes, except DQ, which is padded with 16s"""
    ad.operate(np.pad, padding, 'constant', constant_values=0)
    # We want the mask padding to be DQ.no_data, so have to fix
    # that up by hand...
    if ad[0].mask is not None:
        ad[0].mask[:padding[0][0]] = DQ.no_data
        if padding[0][1] > 0:
            ad[0].mask[-padding[0][1]:] = DQ.no_data
        ad[0].mask[:,:padding[1][0]] = DQ.no_data
        if padding[1][1] > 0:
            ad[0].mask[:,-padding[1][1]:] = DQ.no_data
    if hasattr(ad[0], 'OBJMASK'):
        ad[0].OBJMASK = np.pad(ad[0].OBJMASK, padding, 'constant',
                             constant_values=0)

def _transform_mask(mask, **kwargs):
    """
    Transform the DQ plane, bit by bit. Since np.unpackbits() only works
    on uint8 data, we have to do this by hand
    """
    trans_mask = np.zeros(kwargs['output_shape'], dtype=np.uint16)
    for j in range(0, 16):
        bit = 2**j
        # Only transform bits that have a pixel set. But we always want
        # to do one transformation so we can pad the data with DQ.no_data
        if bit==DQ.no_data or np.sum(mask & bit) > 0:
            temp_mask = affine_transform((mask & 2**j).astype(np.float32),
                                         cval=DQ.no_data if bit==DQ.no_data
                                         else 0, **kwargs)
            trans_mask += np.where(np.abs(temp_mask>0.01*bit), bit,
                                          0).astype(np.uint16)
    return trans_mask
