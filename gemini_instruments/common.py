"""
Structures and functions that can be shared across instrument code.

When should you add something here? Here's some criteria: the code
that belongs in here is not generally useful outside of
gemini_instruments, and:

    - more than one instrument share the code to override a method
      that provides *default* behaviour, meaning that there's no
      common ancestor where you can put this; or,
    - it doesn't make sense to put the code in a method, as it
      doesn't rely on internal knowledge of a class.
"""

__all__ = ['Section']

from collections import namedtuple

from .gmu import sectionStrToIntList

Section = namedtuple('Section', 'x1 x2 y1 y2')

def section_to_tuple(section):
    """
    Takes a string describing a section in the raw format found on
    headers ("[x1:x2,y1:y2]"), and returns a `Section` named tuple
    with the values as integers.

    Parameters
    ----------
    section: str
             The section (in the form [x1:x2,y1:y2]) to be converted to a tuple

    Returns
    -------
    An instance of `Section`
    """

    return Section(*sectionStrToIntList(section))

def build_ir_section(instance, pretty=False):
    """
    1st gen Gemini IR instruments don't have handy BIASSEC/DATASEC/etc keywords.
    This function creates section info out of other keywords.

    Code common to NIRI and GNIRS.

    Parameters
    ----------
    instance: AstroData
        The object that we want to extract the section from
    pretty: bool
        if True, return a string rather than a Section

    Returns
    -------
    A list of `Section` instances.
    """
    # This is identical to the GNIRS code
    hirows = instance.hdr.HIROW
    lowrows = instance.hdr.LOWROW
    hicols = instance.hdr.HICOL
    lowcols = instance.hdr.LOWCOL

    # NOTE: Rows are X and cols are Y? These Romans are crazy
    def format_section(x1,x2,y1,y2, pretty):
        return "[{:d}:{:d},{:d}:{:d}]".format(x1+1, x2+1, y1+1,
            y2+1) if pretty else Section(x1, x2+1, y1, y2+1)
    try:
        xsize = [x2 - x1 + 1 for x1, x2 in zip(lowrows, hirows)]
        ysize = [y2 - y1 + 1 for y1, y2 in zip(lowcols, hicols)]
        return [format_section(512-xs/2, 512+xs/2, 512-ys/2, 512+ys/2, pretty)
                for xs, ys in zip(xsize, ysize)]
    except TypeError:
        xs = hirows - lowrows + 1
        ys = hicols - lowcols + 1
        return format_section(512-xs/2, 512+xsize/2, 512-ys/2, 512+ys/2, pretty)
