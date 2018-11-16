import re
from copy import deepcopy
from datetime import datetime

import astrodata
import gemini_instruments


def isclose(a, b, rel_tol=1e-02, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def expr_parser(expression, strict=False):
    """
    Takes a selection expression and return a codified version of that
    expression that returns True or False when eval().

    Parameters
    ----------
    expression

    Returns
    -------
    String that returns True or False when eval()

    """

    adexpr = re.sub('([_A-z]\w*)([!=<>]+\S+)', r'ad.\1()\2', expression)
    codified_expression = deepcopy(adexpr)

    for strfound in re.finditer('(ad.)([_A-z]\w*)([\(\)]+)([!=<>]+)(\S+)',
                                adexpr):
        descriptor = strfound.groups()[1]
        operator = strfound.groups()[3]
        pattern = r'(ad.' + re.escape(descriptor) + r')([\(\)]+)([!=<>]+)(\S+)'
        if descriptor in ['ut_time', 'local_time']:
            codified_expression = \
                re.sub(pattern,
                       r'\1\2\3datetime.strptime(\4, "%H:%M:%S").time()',
                       codified_expression)
        elif descriptor == 'ut_date':
            codified_expression = \
                re.sub(pattern,
                       r'\1\2\3datetime.strptime(\4, "%Y-%m-%d").date()',
                       codified_expression)
        elif descriptor == 'ut_datetime':
            codified_expression = \
                re.sub(pattern,
                       r'\1\2\3datetime.strptime(\4, "%Y-%m-%d %H:%M:%S")',
                       codified_expression)
        elif descriptor == 'exposure_time' and operator == '==' and not strict:
            codified_expression = \
                re.sub(pattern, r'isclose(\1(),\4)', codified_expression)
        elif descriptor == 'filter_name' and not strict:
            codified_expression = \
                re.sub(pattern, r'\1(pretty=True)\3\4', codified_expression)
        else:
            pass

    return codified_expression

def evalexpression(ad, expression):
    if type(eval(expression)) is not type(True):
        raise(IOError, 'Expression does not return a boolean value.')
    return eval(expression)

def select_data(inputs, tags=[], xtags=[], expression='True'):

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
