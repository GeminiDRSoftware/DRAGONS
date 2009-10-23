 
class PrimitiveParameter( object ):
    '''
    
    
    '''
    name = None
    value = None
    overwrite = None
    help = ""
    
    def __init__(self, name, value=None, overwrite=False, help=""):
        self.name = name
        self.value = value
        self.overwrite = overwrite
        self.help = help

    def __str__(self):
        retstr = "Primitive Parameter (" + str(self.name) + '):'
        retstr = retstr + \
"""
Help      : %(help)s
Value     : %(value)s
Overwrite : %(overwrite)s
""" % {'value':str(self.value), 'overwrite':str(self.overwrite), 'help':str(self.help)}
        return retstr