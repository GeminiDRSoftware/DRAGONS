 
class PrimitiveParameter( object ):
    '''
    
    
    '''
    name = None
    value = None
    overwrite = None
    help = ""
    
    def __init__(self, name, overwrite=False, helps="", value=None):
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
""" % {'value':str(self.value), 'overwrite':str(self.overwrite), 'help':str(self.helps)}
        return retstr