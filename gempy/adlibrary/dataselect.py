import re
from copy import deepcopy
from datetime import datetime  # leave it in.  It is used by the eval.

from importlib import import_module

import astrodata
import gemini_instruments


def isclose(a, b, rel_tol=1e-02, abs_tol=0.0):
    """

    Parameters
    ----------
    a : int/float
        first value
    b : int/float
        second value
    rel_tol : float, (optional)
        relative tolerance
    abs_tol : float, (optional)
        the absolute tolerance

    Returns
    -------

    """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def expr_parser(expression, strict=False):
    """
    Takes a selection expression and return a codified version of that
    expression that returns True or False when eval().

    Parameters
    ----------
    expression : str
        the string that is parsed by the function
    strict : Bool
        Defines if expr_parser is "strict" or not

    Returns
    -------
    String that returns True or False when eval()

    """

    adexpr = re.sub(r'([_A-z]\w*)([!=<>]+\S+)', r'ad.\1()\2', expression)
    codified_expression = ''
    components = re.split(r'(\s+and\s+|\s+or\s+|\s+not\s+)', adexpr)

    for item in components:
        match = re.match(r'(ad.)([_A-z]\w*)([\(\)]+)([!=<>]+)(\S+)', item)
        if match:
            descriptor = match.groups()[1]
            operator = match.groups()[3]
            pattern = r'(ad.' + re.escape(
                descriptor) + r')([\(\)]+)([!=<>]+)(\S+)'
            if descriptor in ['ut_time', 'local_time']:
                item = re.sub(r'\)$', ' )', item)
                codified_item = \
                    re.sub(pattern,
                           r'\1\2\3datetime.strptime(\4, "%H:%M:%S").time()',
                           item)
            elif descriptor == 'ut_date':
                item = re.sub(r'\)$', ' )', item)
                codified_item = \
                    re.sub(pattern,
                           r'\1\2\3datetime.strptime(\4, "%Y-%m-%d").date()',
                           item)
            elif descriptor == 'ut_datetime':
                item = re.sub(r'\)$', ' )', item)
                codified_item = \
                    re.sub(pattern,
                           r'\1\2\3datetime.strptime(\4, "%Y-%m-%d %H:%M:%S")',
                           item)
            elif descriptor == 'exposure_time' and operator == '==' and not strict:
                codified_item = \
                    re.sub(pattern, r'isclose(\1(),\4)', item)
            elif descriptor == 'central_wavelength' and operator == '==' and \
                not strict:
                item = re.sub(r'\)$', ' )', item)
                codified_item = \
                    re.sub(pattern, r'isclose(\1(),\4, rel_tol=1e-5)', item)
            elif descriptor in ['filter_name', 'detector_name', 'disperser'] \
                and not strict:
                codified_item = re.sub(pattern, r'\1(pretty=True)\3\4', item)
            else:
                codified_item = item

            codified_expression += codified_item
        else:
            codified_expression += item

    return codified_expression


def evalexpression(ad, expression):
    """
    Tests to make sure that the expression passed from the user is a Bool
    Parameters
    ----------
    ad : astrodata.object
        Unused, remove if necessary
    expression : str
        expression from user

    Returns
    -------
    expression - bool
        evaluated expression, True or False
    """
    if type(eval(expression)) is not type(True):
        raise IOError('Expression does not return a boolean value.')
    return eval(expression)


def select_data(inputs, tags=[], xtags=[], expression='True', adpkg=None):
    """
    Given a list of fits files, function will return a list of astrodata objects
    that satisfy all the parameters

    Parameters
    ----------
    inputs - list of strings
        list of paths with full directory to a fits file
    tags - list of strings
        list with all the tags you want to make sure are in the inputs fits files
    xtags - list of strings
        list with all the tags you DO NOT want in the astrodata objects
    expression - usually dataselect.expr_parser object
        filters for any specific constraining like data,
        datetime, filter name, exposure time

    Returns
    list of astrodata objects
        returns all objects that satisfy all the above requirements
    -------
    """
    if adpkg is not None:
        import_module(adpkg)

    selected_data = []
    for input in inputs:
        ad = astrodata.open(input)
        adtags = ad.tags
        if set(tags).issubset(adtags) and \
               not len(set(xtags).intersection(adtags)) and \
               evalexpression(ad, expression):
           selected_data.append(input)

    return selected_data

def writeheader(fh, tags, xtags, expression):
    """
    Given a list of fits files, function will return a list of astrodata objects
    that satisfy all the parameters

    Parameters
    ----------
    fh -
        opened file
    tags - list of strings
        list with all the tags you want to make sure are in the inputs fits files
    xtags - list of strings
        list with all the tags you DO NOT want in the astrodata objects
    expression - usually dataselect.expr_parser object
        filters for any specific constraining like data,
        datetime, filter name, exposure time

    Returns
    N/a
    -------
    """
    if tags is None:
        tags = 'None'
    if xtags is None:
        xtags = 'None'
    if expression is None:
        expression = 'None'
    fh.write('# Includes tags: '+str(tags)+'\n')
    fh.write('# Excludes tags: '+str(xtags)+'\n')
    fh.write('# Descriptor expression: '+expression+'\n')
    return
