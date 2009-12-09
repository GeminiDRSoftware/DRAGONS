
_displayObj = None


class DisplayException:
    pass

def getDisplay():
    global _displayObj
    
    if _displayObj is None:
        _displayObj = display()
    
    return _displayObj


class display:
    
    def __init__(self):
        self.displayDict = {}
        
    def __addtool__(self, tool, id=None):
        tooltype = str(type(tool))
        
        if not self.displayDict.has_key( tooltype ):
            self.displayDict.update( tooltype, [] )
            
        self.displayDict[tooltype].append(DS9)
        
        if id is not None:
            self.displayDict.update( id, tool)
    
    def __getitem__(self, id):
        return self.displayDict[id]
    
    def displayTool(self, tool, id=None, **toolargs ):
        '''
        
        
        '''
        if type( tool ) == str:
            toolMethod = 'create'+tool.upper()
            if self.hasattr( toolMethod ):
                toolCreator = self.getattr( toolMethod )
                newtool = toolCreator( **toolargs )
                self.__addtool__( newtool, id )
            else:
                raise DisplayException( 'The display program: "%s" is not supported.' %(tool) )
        else:
            raise DisplayException( 'Invalid Argument type: "%s", must be str' %( str(type(tool)) ) )
    
    
    
    
    
    def createDS9(self, ds9type=True, **args):
        if ds9type:
            try:
                import pysao
            except:
                print 'Unable to import pysao.'
                return self.createDS9( ds9type=False, **args )
            return DS9( **args )
        else:
            try:
                return DS9( ds9type='iraf', **args )
            except:
                raise
            

class DS9:
    
    def __init__(self, ds9type='pysao', **args):
        self.ds9type = ds9type
        
        
        if self.ds9type == 'pysao':
            try:
                import pysao
            except:
                raise DisplayException( 'Cannot import pysao.' )
            self = pysao.ds9( **args )
        else:
            try:
                import pyraf
                from pyraf import iraf
            except:
                raise DisplayException( 'Cannot import pyraf.' )
            
    def loadFile(self, filename, frame):
        if self.ds9type == 'pysao':
            pass
        else:
            pass
        
    def findFile(self, filename):
        pass
        
            