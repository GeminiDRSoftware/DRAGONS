import numpy as np
import math


def make_overlay_mask(overlay, shape):
    """
    Generates a tuple of numpy arrays that can be used to mask a display with
    circles centered on given positions, e.g., data[iqmask] = some_value

    The circle definition is based on numdisplay.overlay.circle, but circles
    are two pixels wide to make them easier to see.

    Parameters
    ----------
    overlay : list of 3-tuples (xcenter, ycenter, radius): 1-indexed

    shape : 2-tuple
        shape of the data being displayed

    Returns
    -------
    tuple: 0-indexed arrays of y and x coordinates for overlay
    """
    xind, yind = [], []
    height, width = shape
    for x0, y0, radius in overlay:
        if radius == 0:
            # Convert from 1-indexed to 0-indexed
            xind.append(int(x0 - 0.5))
            yind.append(int(y0 - 0.5))
            continue

        r2 = radius * radius
        quarter = int(math.ceil(radius * math.sqrt(0.5)))
        for dy in range(-quarter, quarter+1):
            j = int(round(y0 + dy))
            if 0 < j <= height:
                dx = math.sqrt(r2 - dy**2) if r2 > dy*dy else 0
                i = int(round(x0 - dx))           # left arc
                xind.extend([i-1, i-2])
                yind.extend([j-1, j-1])
                i = int(round(x0 + dx))           # right arc
                xind.extend([i-1, i])
                yind.extend([j-1, j-1])

        for dx in range(-quarter, quarter+1):
            dy = math.sqrt(r2 - dx**2) if r2 > dx*dx else 0
            i = int(round(dx + x0))
            if 0 < i <= width:
                j = int(round(y0 - dy))           # bottom arc
                xind.extend([i-1, i-1])
                yind.extend([j-1, j-2])
                j = int(round(y0 + dy))           # top arc
                xind.extend([i-1, i-1])
                yind.extend([j-1, j])

    xind = np.array(xind)
    yind = np.array(yind)
    # I've taken the decision to do minimal if...then checking when making
    # the (xind, yind) lists, and removing all off-image pixels here with
    # numpy. This ought to be faster.
    on_image = np.logical_and.reduce([xind >= 0, xind < width,
                                      yind >= 0, yind < height])
    iqmask = (yind[on_image], xind[on_image])
    return iqmask
