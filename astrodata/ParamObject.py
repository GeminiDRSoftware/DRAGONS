#
#                                                                 gemini_python
#
#                                                                      astrodata
#                                                                 ParamObject.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Rev$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
class PrimitiveParameter(object):
    """
    This is the object which contains all relevant variables where primitive 
    parameter data are stored.
    """
    def __init__(self, name, value=None, overwrite=False, helps=""):
        self.name = name
        self.value = value
        self.overwrite = overwrite
        self.helps = helps

    def __str__(self):
        retstr = "Parameter (" + str(self.name) + '):'
        retstr = retstr + \
                 """
                 Help      : %(help)s
                 Value     : %(value)s
                 Overwrite : %(overwrite)s
                 """ % {'value':str(self.value), 
                        'overwrite':str(self.overwrite), 
                        'help':str(self.helps)}
        return retstr
