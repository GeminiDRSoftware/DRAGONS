import numpy as np
from gempy.utils import logutils

# ------------------------------------------------------------------------------
def remove_single_length_dimension(adinput):
    """
    If there is only one single length dimension in the pixel data, the
    remove_single_length_dimension function will remove the single length
    dimension. In addition, this function removes any keywords associated with
    that dimension. Used ONLY by the standardizeStructure primitive in
    primitives_F2.py.

    Parameters
    ----------
    adinput
        AstroDataFits: input AD object

    Returns
    -------
        AstroData: modified object of same subclass as input
    """

    log = logutils.get_logger(__name__)

    for ext in adinput:
        # Ensure that there is only one single length dimension in the pixel
        # data
        if ext.data.shape.count(1) == 1:

            # Determine the position of the single length dimension in the
            # tuple of array dimensions output by ext.data.shape
            axis = np.where([length==1 for length in ext.data.shape])[0][0]

            # numpy arrays use 0-based indexing and the axes are ordered from
            # slow to fast. So, if the position of the single length dimension
            # is located in e.g., ext.data.shape[0], the dimension number of
            # the FITS pixel data array is ext.data.ndim + 1 (since FITS pixel
            # data arrays use 1-based indexing).
            dimension = ext.data.ndim - axis

            # The np.squeeze method only removes a dimension from the array if
            # the dimension has a length equal to 1
            log.status("Removing dimension {} from {}".
                       format(dimension, adinput.filename))
            ext.operate(np.squeeze)

            # Set the NAXIS keyword appropriately now that a dimension has been
            # removed
            ext.hdr.set("NAXIS", ext.data.ndim)

            # This should be a log.debug call, but that doesn't appear to work
            # right now, so using log.fullinfo
            #log.fullinfo("Updated dimensions of {}[{},{}] = {}".format(adinput.filename,
            #                                                       ext.hdr['EXTNAME'],
            #                                                       ext.hdr['EXTVER'],
            #                                                       ext.data.shape))

            # Remove the keywords relating to the dimension that has been
            # removed (IRAF seems to add WCSDIM=3, CTYPE3='LINEAR  ', CD3_3=1.,
            # LTM1_1=1., LTM2_2=1., LTM3_3=1., WAXMAP01='1 0 2 0 0 0 ',
            # WAT0_001='system=image', WAT1_001='wtype=tan axtype=ra' and
            # WAT2_001= 'wtype=tan axtype=dec' when doing e.g., imcopy
            # f2.fits[*,*,1], so perhaps these should be removed as well?)
            #
            # old data don't have all keywords.  need to check first.
            keywords = ("NAXIS{0}, AXISLAB{0}, CD{0}_{0}".format(dimension))
            for keyword in keywords.split(','):
                if keyword in ext.hdr:
                    ext.hdr.remove(keyword)
        else:
            log.warning("No dimension of length 1 in extension pixel data."
                        "No changes will be made to {}. ".format(adinput.filename))

    return adinput
