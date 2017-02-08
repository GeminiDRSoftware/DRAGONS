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

def build_group_id(ad, desc_list, prettify=(), force_list=(), additional=None):
    """
    Builds a Group ID from information found in the descriptors. It takes a number
    of descriptor names, invokes and then concatenates their result (converted to string)
    to from a group ID.  Additional parameters can be passed to modify the result.

    Parameters
    ----------
    ad: AstroData
              An instance of `AstroData` derivative that the descriptors will be

    desc_list: list of str
              A list of descriptor names (order matters) which will be used to
              build the Group ID

    prettify: sequence/set of str
              Names of descriptors that need to be invoked with `pretty=True`

    force_list: sequence/set of str
              The descriptors named in this list will have their results coerced
              into a list, if they returned something else.

    additional: str
              Additional information that will be added verbatim at the end of
              the Group ID

    Returns
    -------
    A string with the group id
    """
    desc_object_string_list = []
    for descriptor in desc_list:
        desc_method = getattr(ad, descriptor)
        if descriptor in prettify or 'section' in descriptor:
            desc_object = desc_method(pretty=True)
        else:
            desc_object = desc_method()

        # Ensure we get a list, even if only looking at one extension
        if (descriptor in force_list and
                not isinstance(desc_object, list)):
            desc_object = [desc_object]

        # Convert descriptor to a string and store
        desc_object_string_list.append(str(desc_object))

    # Add in any none descriptor related information
    if additional is not None:
        desc_object_string_list.append(additional)

    # Create and return the final group_id string
    return '_'.join(desc_object_string_list)

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

def build_ir_section(ad, pretty=False):
    """
    1st gen Gemini IR instruments don't have handy BIASSEC/DATASEC/etc keywords.
    This function creates section info out of other keywords.

    Code common to NIRI and GNIRS.

    Parameters
    ----------
    ad: AstroData
        The object that we want to extract the section from
    pretty: bool
        if True, return a string rather than a Section

    Returns
    -------
    A list of `Section` instances.
    """
    # This is identical to the GNIRS code
    try:
        hirows = ad.hdr['HIROW']
        lowrows = ad.hdr['LOWROW']
        hicols = ad.hdr['HICOL']
        lowcols = ad.hdr['LOWCOL']
    except KeyError:
        return None

    # NOTE: Rows are X and cols are Y? These Romans are crazy
    def format_section(x1,x2,y1,y2, pretty):
        return "[{:d}:{:d},{:d}:{:d}]".format(x1+1, x2+1, y1+1,
            y2+1) if pretty else Section(x1, x2+1, y1, y2+1)

    if ad.is_single:
        xs = hirows - lowrows + 1
        ys = hicols - lowcols + 1
        return format_section(512 - xs // 2, 512 + xs // 2 - 1,
                              512 - ys // 2, 512 + ys // 2 - 1, pretty)
    else:
        xsize = [x2 - x1 + 1 for x1, x2 in zip(lowrows, hirows)]
        ysize = [y2 - y1 + 1 for y1, y2 in zip(lowcols, hicols)]
        return [format_section(512 - xs // 2, 512 + xs // 2 - 1,
                               512 - ys // 2, 512 + ys // 2 - 1, pretty)
                for xs, ys in zip(xsize, ysize)]
