from astrodata import wcs as adwcs
from gempy.utils import logutils


def remove_single_length_dimension(adinput):
    """
    If there is only one single length dimension in the pixel data, the
    remove_single_length_dimension function will remove the single length
    dimension. In addition, this function removes any keywords associated with
    that dimension. Used ONLY by the standardizeStructure primitive in
    primitives_F2.py.

    Parameters
    ----------
    adinput : AstroData
        input AD object

    Returns
    -------
        AstroData: modified object of same subclass as input
    """

    log = logutils.get_logger(__name__)

    for ext in adinput:
        # Ensure that there is only one single length dimension in the pixel
        # data
        shape = ext.shape
        extid = f"{adinput.filename}:{ext.id}"
        if shape.count(1) == 1:
            axis = shape.index(1)

            # Dimension in user-friendly format (1=x, 2=y, etc.)
            dimension = ext.data.ndim - axis

            log.debug(f"Removing dimension {dimension} from {extid}")
            _slice = tuple(0 if i == axis else slice(None)
                           for i, _ in enumerate(shape))
            ext.reset(ext.nddata[_slice])
            try:
                adwcs.remove_unused_world_axis(ext)
            except:
                log.debug(f"Cannot remove world axis from {extid}")
        else:
            log.warning("No dimension of length 1 in extension pixel data."
                        f"No changes will be made to {extid}")

    return adinput
