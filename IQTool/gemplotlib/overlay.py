import math
import struct

import numpy as N
import numdisplay

"""The public functions are the following.  For point, rectangle, circle
and polyline, arguments shown on separate lines are alternate ways to
specify the location and size of the figure to be drawn.

point (x=x0, y=y0,
       center=(x0,y0))
rectangle (left=x1, right=x2, lower=y1, upper=y2,
           center=(x0,y0), width=w, height=h)
circle (x=x0, y=y0, radius=r,
        center=(x0,y0), radius=r)
polyline (points=[(x1,y1), (x2,y2), (x3,y3), ...],
          vertices=[(x1,y1), (x2,y2), (x3,y3), ...])
undo()
set (color, radius)

color is an optional argument to point, rectangle, circle, and polyline.
The allowed values for color are:
C_BLACK, C_WHITE, C_RED, C_GREEN, C_BLUE, C_YELLOW, C_CYAN, C_MAGENTA,
C_CORAL, C_MAROON, C_ORANGE, C_KHAKI, C_ORCHID, C_TURQUOISE, C_VIOLET, C_WHEAT
"""

C_BLACK     = 202
C_WHITE     = 203
C_RED       = 204
C_GREEN     = 205
C_BLUE      = 206
C_YELLOW    = 207
C_CYAN      = 208
C_MAGENTA   = 209
# the following all seem to display as magenta when using ds9
C_CORAL     = 210
C_MAROON    = 211
C_ORANGE    = 212
C_KHAKI     = 213
C_ORCHID    = 214
C_TURQUOISE = 215
C_VIOLET    = 216
C_WHEAT     = 217

# These two are used for saving the displayed values before drawing
# an overlay, to allow restoring the display (via undo).
global_save = []
global_byte_buf = N.zeros (1, dtype=N.uint8)

# These two are for convenience, so they can take default values rather
# than having to be specified for each function call.  The radius is
# only relevant for circles.  Either or both of these can be set via
# the set() function.
global_color = N.array ((C_CYAN,), dtype=N.uint8)
global_radius = 3

def _open_display(frame=1):
    """Open the device."""

    fd = numdisplay.displaydev._open()
    fd.setFrame(frame_num=frame)
    (tx, ty, fbwidth, fbheight) = fd.readInfo()
    return (fd, tx, ty, fbwidth, fbheight)

def set (color=None, radius=None):
    """Specify the color or the radius.

    @param color: color code to use for graphic overlays; the allowed
        values (202..217) are:
        C_BLACK, C_WHITE, C_RED, C_GREEN, C_BLUE, C_YELLOW, C_CYAN, C_MAGENTA,
        C_CORAL, C_MAROON, C_ORANGE, C_KHAKI, C_ORCHID, C_TURQUOISE, C_VIOLET,
        C_WHEAT
    @type color: int
    @param radius: radius to use when drawing circles
    @type radius: int
    """

    global global_color, global_radius

    if color is not None:
        global_color = _checkColor (color)
    if radius is not None:
        if radius < 0:
            raise ValueError, "radius must be non-negative"
        global_radius = radius


def _transformPoint (x, y, tx, ty):
    """Convert image pixel coords to frame buffer coords (IIS convention).

    @param x: image X coordinate of point
    @type x: float
    @param y: image Y coordinate of point
    @type y: float
    @param tx: X offset of frame buffer in image (positive if image is larger)
    @type tx: float
    @param ty: image coordinate corresponding to top line of frame buffer
    @type ty: float
    """

    x -= tx
    y = ty - y
    return (x, y)

def _checkColor (color=None):
    """Return a valid color.

    @param color: color code to use; if color=None, use default
    @type color: int
    """

    global global_color

    if color is None:
        color = global_color
    elif color < C_BLACK or color > C_WHEAT:
        raise ValueError, "%d is not a valid color" % color
    else:
        color = N.array ((color,), dtype=N.uint8)
    return color

def _update_save (fd, x, y, list_of_points, last_overlay):
    """Save info in local lists list_of_points and last_overlay.

    @param fd: for reading from image display
    @type fd: file handle
    @param x: X coordinate (IIS convention, not image coordinates)
    @type x: int
    @param y: Y coordinate (IIS convention, not image coordinates)
    @type y: int
    @param list_of_points: pixels that have been written to by the graphic
        overlay that called this function (updated by this function)
    @type list_of_points: list of (x,y) tuples
    @param last_overlay: pixel coordinates and current value of display
        (updated by this function)
    @type last_overlay: list of (x,y,value) tuples
    """

    global global_byte_buf

    if (x, y) not in list_of_points:
        value = fd.readData (x, y, global_byte_buf)
        value = struct.unpack ('B', value)
        list_of_points.append ((x, y))
        last_overlay.append ((x, y, value[0]))

def point (**kwargs):
    """Draw a point.

    @param x: image X coordinate of point
    @type x: int
    @param y: image Y coordinate of point
    @type y: int
    @param center: (x,y) coordinates of point
    @type center: tuple
    @param color: color code to use; if not specified, use default
    @type color: int

    syntax:
        overlay.point (x=x0, y=y0)
        overlay.point (center=(x0,y0))
        overlay.point (x=x0, y=y0, color=overlay.C_<color>)
    """

    # These are used for saving what is currently displayed, for use by
    # the undo() function.
    global global_save, global_byte_buf
    last_overlay = []
    list_of_points = []

    allowed_arguments = ["x", "y", "center", "color", "frame"]
    x = None; y = None; center = None; color = None; frame = None
    keys = kwargs.keys()
    if "center" in keys and ("x" in keys or "y" in keys):
        raise ValueError, \
            "Specify either 'center' or 'x' and 'y', but not both."
    for key in keys:
        if key in allowed_arguments:
            if key == "center":
                (x, y) = kwargs["center"]
            elif key == "x":
                x = kwargs["x"]
            elif key == "y":
                y = kwargs["y"]
            elif key == "color":
                color = kwargs["color"]
            elif key == "frame":
                frame = kwargs["frame"]
        else:
            raise ValueError, \
            "Invalid argument to 'point'; use 'x', 'y', 'center', 'color' or 'frame'."
    if x is None or y is None:
        raise ValueError, "You must specify either 'x' and 'y' or 'center'."

    color = _checkColor (color)

    (fd, tx, ty, fbwidth, fbheight) = _open_display(frame=frame)

    (x, y) = _transformPoint (x, y, tx, ty)
    if x >= 0 and y >= 0 and x < fbwidth and y < fbheight:
        # save the value that is currently at (x,y)
        _update_save (fd, x, y, list_of_points, last_overlay)
        # write a new value at (x,y)
        fd.writeData (x, y, color)

    global_save.append (last_overlay)

    fd.close()

def rectangle (**kwargs):
    """Draw a rectangle.

    @param left: image X coordinate of left edge
    @type left: int
    @param right: image X coordinate of right edge
    @type right: int
    @param lower: image Y coordinate of lower edge
    @type lower: int
    @param upper: image Y coordinate of upper edge
    @type upper: int
    @param center: (x,y) coordinates of middle of rectangle
    @type center: tuple
    @param width: width of rectangle (X direction)
    @type width: int
    @param height: height of rectangle (Y direction)
    @type height: int
    @param color: color code to use; if not specified, use default
    @type color: int

    syntax:
        overlay.rectangle (left=x1, right=x2, lower=y1, upper=y2)
        overlay.rectangle (center=(x0,y0), width=w, height=h)
        overlay.rectangle (left=x1, lower=y1, center=(x0,y0))
        overlay.rectangle (left=x1, lower=y1, width=w, height=h)
        overlay.rectangle (right=x2, upper=y2, width=w, height=h)
        overlay.rectangle (right=x2, upper=y2, center=(x0,y0))
        overlay.rectangle (left=x1, right=x2, lower=y1, upper=y2,
                           color=overlay.C_<color>)
    """

    # These are used for saving what is currently displayed, for use by
    # the undo() function.
    global global_save, global_byte_buf
    last_overlay = []
    list_of_points = []

    allowed_arguments = ["left", "right", "lower", "upper",
                         "center", "width", "height", "color"]
    x1 = None; x2 = None; y1 = None; y2 = None
    center = None; width = None; height = None; color = None

    error_message = "Invalid argument(s) to 'rectangle'; use one of:\n" \
"  left=x1, right=x2, lower=y1, upper=y2, or\n" \
"  center=(x0,y0), width=w, height=h, or\n" \
"  left=x1, lower=y1, width=w, height=h, or\n" \
"  right=x2, upper=y2, width=w, height=h\n" \
"  color may also be specified."

    keys = kwargs.keys()
    if "center" in keys:
        if "width" not in keys or "height" not in keys:
            raise ValueError, error_message
    if "left" in keys:
        if "right" not in keys and "width" not in keys:
            raise ValueError, error_message
    if "lower" in keys:
        if "upper" not in keys and "height" not in keys:
            raise ValueError, error_message
    if "right" in keys:
        if "left" not in keys and "width" not in keys:
            raise ValueError, error_message
    if "upper" in keys:
        if "lower" not in keys and "height" not in keys:
            raise ValueError, error_message

    for key in keys:
        if key in allowed_arguments:
            if key == "center":
                center = kwargs["center"]
                if not isinstance (center, (list, tuple)):
                    raise ValueError, error_message
                (x0, y0) = center
            elif key == "left":
                x1 = kwargs["left"]
            elif key == "right":
                x2 = kwargs["right"]
            elif key == "lower":
                y1 = kwargs["lower"]
            elif key == "upper":
                y2 = kwargs["upper"]
            elif key == "width":
                width = kwargs["width"]
            elif key == "height":
                height = kwargs["height"]
            elif key == "color":
                color = kwargs["color"]
        else:
            raise ValueError, error_message

    if x1 is None:
        if center is not None and width is not None:
            x1 = int (round (x0 - width/2.))
        elif x2 is not None and width is not None:
            x1 = x2 - width
    if x2 is None:
        if center is not None and width is not None:
            x2 = int (round (x0 + width/2.))
        elif x1 is not None and width is not None:
            x2 = x1 + width
    if y1 is None:
        if center is not None and height is not None:
            y1 = int (round (y0 - height/2.))
        elif y2 is not None and height is not None:
            y1 = y2 - height
    if y2 is None:
        if center is not None and height is not None:
            y2 = int (round (y0 + height/2.))
        elif y1 is not None and height is not None:
            y2 = y1 + height

    if x1 is None or x2 is None or y1 is None or y2 is None:
        raise ValueError, error_message

    color = _checkColor (color)

    (fd, tx, ty, fbwidth, fbheight) = _open_display()

    (x1, y1) = _transformPoint (x1, y1, tx, ty)
    (x2, y2) = _transformPoint (x2, y2, tx, ty)

    if x2 < x1:
        (x1, x2) = (x2, x1)
    if y2 < y1:
        (y1, y2) = (y2, y1)

    imin = max (0, x1)
    imax = min (x2+1, fbwidth)
    if y1 >= 0 and y1 < fbheight:
        for i in range (imin, imax):
            _update_save (fd, i, y1, list_of_points, last_overlay)
            fd.writeData (i, y1, color)
    if y2 >= 0 and y2 < fbheight:
        for i in range (imin, imax):
            _update_save (fd, i, y2, list_of_points, last_overlay)
            fd.writeData (i, y2, color)

    jmin = max (0, y1)
    jmax = min (y2+1, fbheight)
    if x1 >= 0 and x1 < fbwidth:
        for j in range (jmin, jmax):
            _update_save (fd, x1, j, list_of_points, last_overlay)
            fd.writeData (x1, j, color)
    if x2 >= 0 and x2 < fbwidth:
        for j in range (jmin, jmax):
            _update_save (fd, x2, j, list_of_points, last_overlay)
            fd.writeData (x2, j, color)

    global_save.append (last_overlay)

    fd.close()

def circle (**kwargs):
    """Draw a circle.

    @param x: image X coordinate of center
    @type x: int
    @param y: image Y coordinate of center
    @type y: int
    @param center: (x,y) coordinates of center
    @type center: tuple
    @param radius: radius of circle
    @type radius: int
    @param color: color code to use; if not specified, use default
    @type color: int

    syntax:
        overlay.circle (x=x0, y=y0, radius=r)
        overlay.circle (center=(x0,y0), radius=r)
        overlay.circle (x=x0, y=y0, radius=r, color=overlay.C_<color>)
    """

    # These are used for saving what is currently displayed, for use by
    # the undo() function.
    global global_save, global_byte_buf
    last_overlay = []
    list_of_points = []

    allowed_arguments = ["x", "y", "center", "radius", "color", "frame"]
    x0 = None; y0 = None; center = None;
    radius = global_radius; color = None; frame = None

    error_message = "Invalid argument(s) to 'circle'; use either:\n" \
"  x=x0, y=x0, radius=r, or\n" \
"  center=(x0,y0), radius=r\n" \
"  color and frame may also be specified."

    keys = kwargs.keys()
    if "center" in keys and ("x" in keys or "y" in keys):
        raise ValueError, error_message
    for key in keys:
        if key in allowed_arguments:
            if key == "center":
                center = kwargs["center"]
                if not isinstance (center, (list, tuple)):
                    raise ValueError, error_message
                (x0, y0) = center
            elif key == "x":
                x0 = kwargs["x"]
            elif key == "y":
                y0 = kwargs["y"]
            elif key == "radius":
                radius = kwargs["radius"]
            elif key == "color":
                color = kwargs["color"]
            elif key == "frame":
                frame = kwargs["frame"]
        else:
            raise ValueError, error_message
    if x0 is None or y0 is None or radius is None:
        raise ValueError, error_message

    color = _checkColor (color)

    (fd, tx, ty, fbwidth, fbheight) = _open_display(frame=frame)

    (x0, y0) = _transformPoint (x0, y0, tx, ty)
    quarter = int (math.ceil (radius * math.sqrt (0.5)))
    r2 = radius**2

    for dy in range (-quarter, quarter+1):
        dx = math.sqrt (r2 - dy**2)
        j = int (round (dy + y0))
        i = int (round (x0 - dx))           # left arc
        if i >= 0 and j >= 0 and i < fbwidth and j < fbheight:
            _update_save (fd, i, j, list_of_points, last_overlay)
            fd.writeData (i, j, color)
        i = int (round (x0 + dx))           # right arc
        if i >= 0 and j >= 0 and i < fbwidth and j < fbheight:
            _update_save (fd, i, j, list_of_points, last_overlay)
            fd.writeData (i, j, color)

    for dx in range (-quarter, quarter+1):
        dy = math.sqrt (r2 - dx**2)
        i = int (round (dx + x0))
        j = int (round (y0 - dy))           # bottom arc
        if i >= 0 and j >= 0 and i < fbwidth and j < fbheight:
            _update_save (fd, i, j, list_of_points, last_overlay)
            fd.writeData (i, j, color)
        j = int (round (y0 + dy))           # top arc
        if i >= 0 and j >= 0 and i < fbwidth and j < fbheight:
            _update_save (fd, i, j, list_of_points, last_overlay)
            fd.writeData (i, j, color)

    global_save.append (last_overlay)

    fd.close()






def polyline (**kwargs):
    """Draw a series of connected line segments.

    @param points: (x,y) points to connect with line segments
    @type points: list of tuples
    @param vertices: (x,y) points to connect with line segments
    @type vertices: list of tuples
    @param color: color code to use; if not specified, use default
    @type color: int

    syntax:
        overlay.polyline (points=[(x1,y1), (x2,y2), (x3,y3)])
        overlay.polyline (vertices=[(x1,y1), (x2,y2), (x3,y3)])
        overlay.polyline (points=[(x1,y1), (x2,y2), (x3,y3)],
                          color=overlay.C_<color>)
    """

    # These are used for saving what is currently displayed, for use by
    # the undo() function.
    global global_save, global_byte_buf
    last_overlay = []
    list_of_points = []

    allowed_arguments = ["points", "vertices", "color", "frame"]
    points = None; vertices = None; color = None; frame = None

    error_message = "Invalid argument(s) to 'polyline'; use either:\n" \
"  points=[(x1,y1), (x2,y2), (x3,y3), <...>] or\n" \
"  vertices=[(x1,y1), (x2,y2), (x3,y3), <...>]\n" \
"  color or frame may also be specified."

    keys = kwargs.keys()
    if "points" not in keys and "vertices" not in keys:
        raise ValueError, error_message
    for key in keys:
        if key in allowed_arguments:
            if key == "points":
                points = kwargs["points"]
                if not isinstance (points, (list, tuple)):
                    raise ValueError, error_message
            elif key == "vertices":
                vertices = kwargs["vertices"]
                if not isinstance (vertices, (list, tuple)):
                    raise ValueError, error_message
            elif key == "color":
                color = kwargs["color"]
            elif key == "frame":
                frame = kwargs["frame"]
        else:
            raise ValueError, error_message

    if points is not None and vertices is not None:
        raise ValueError, error_message
    if vertices is not None:
        keyword = "vertices"
        points = vertices
    else:
        keyword = "points"

    color = _checkColor (color)

    (fd, tx, ty, fbwidth, fbheight) = _open_display(frame=frame)

    expected_a_tuple = \
    "Each point in %s for polyline must be a two-element list or tuple,\n" \
    "giving the X and Y image pixel coordinates of a vertex."

    first = True
    for point in points:
        if not isinstance (point, (list, tuple)):
            raise ValueError, expected_a_tuple
        (x, y) = point
        (x, y) = _transformPoint (x, y, tx, ty)
        if first:
            (xlast, ylast) = (x, y)
            first = False
            continue
        if x == xlast and y == ylast:
            continue
        dx = x - xlast
        dy = y - ylast
        (x1, x2, y1, y2) = (xlast, x, ylast, y)
        if abs (dy) <= abs (dx):
            if x >= xlast:
                step = 1
            else:
                step = -1
            slope = float(dy) / float(dx)
            if step > 0:
                imin = max (0, x1)
                imax = min (x2+1, fbwidth)
            else:
                imin = min (x1, fbwidth-1)
                imax = max (x2-1, -1)
            for i in range (imin, imax, step):
                j = slope * (i - x1) + y1
                j = int (round (j))
                if j >= 0 and j < fbheight:
                    _update_save (fd, i, j, list_of_points, last_overlay)
                    fd.writeData (i, j, color)
        else:
            if y >= ylast:
                step = 1
            else:
                step = -1
            slope = float(dx) / float(dy)
            if step > 0:
                jmin = max (0, y1)
                jmax = min (y2+1, fbheight)
            else:
                jmin = min (y1, fbheight-1)
                jmax = max (y2-1, -1)
            for j in range (jmin, jmax, step):
                i = slope * (j - y1) + x1
                i = int (round (i))
                if i >= 0 and i < fbwidth:
                    _update_save (fd, i, j, list_of_points, last_overlay)
                    fd.writeData (i, j, color)
        xlast = x
        ylast = y

    global_save.append (last_overlay)

    fd.close()

def undo():
    """Restore the values before the last overlay was written."""

    global global_save, global_byte_buf

    if len (global_save) == 0:
        return

    (fd, tx, ty, fbwidth, fbheight) = _open_display()

    last_overlay = global_save.pop()
    for (x, y, value) in last_overlay:
        global_byte_buf[0] = value
        fd.writeData (x, y, global_byte_buf)

    fd.close()
