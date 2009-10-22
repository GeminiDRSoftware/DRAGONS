 
class PrimitiveParameter( object ):
    '''
    
    
    '''
    name = None
    value = None
    overwrite = None
    
    def __init__(self, name, value=None, overwrite=False):
        self.name = name
        self.value = value
        self.overwrite = overwrite

    def __str__(self):
        retstr = "Primitive Parameter (" + str(self.name) + '):'
        retstr = retstr + \
"""
Value     : %(value)s
Overwrite : %(overwrite)s
""" % {'value':str(self.value), 'overwrite':str(self.overwrite)}
        return retstr