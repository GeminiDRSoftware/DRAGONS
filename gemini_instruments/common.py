"""
Structures and functions that can be shared across instrument code
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
    A instance of `Section`
    """

    return Section(*sectionStrToIntList(section))
