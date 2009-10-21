 
class PrimitiveParameter( object ):
    '''
    
    
    '''
    name = None
    overwriteable = None
    
    def __init__(self, name, overwriteable=False):
        self.name = name
        self.overwriteable = overwriteable

    def __str__(self):
        retstr = "Primitive Paremeter: " + str(self.name) + '\n'
        retstr = retstr + \
"""
Overwritable: %(overwriteable)s\n
""" % {'overwriteable':str(self.overwriteable)}
