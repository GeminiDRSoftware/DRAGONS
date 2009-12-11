from utils.future import pyDisplay
import astrodata
from astrodata.AstroData import AstroData
from astrodata import Descriptors
from astrodata import IDFactory
from pyraf import iraf

def supported():
    return pyDisplay.supported()

class DisplayServiceException:
    pass

_displayObj = None

def getDisplayService():
    global _displayObj
    
    if _displayObj is None:
        _displayObj = display()
    
    return _displayObj


class DisplayService(object):
    '''
    
    '''
#------------------------------------------------------------------------------ 
    def __getitem__(self, item):
        return self.ds9.__getitem__(item)
#------------------------------------------------------------------------------ 
    def __setitem__(self, item, value):
        self.ds9.__setitem__(item, value)
#------------------------------------------------------------------------------ 
    def __init__(self):
        self.display = pyDisplay.getDisplay()
        self.ds9 = self.setupDS9()
#------------------------------------------------------------------------------ 
    def setupDS9(self):
        storeDs9 = self.display.getDisplayTool( 'ds9' )
        storeDs9.set( 'tile yes' )
        return storeDs9
#------------------------------------------------------------------------------ 
    def display(self, ad, id=None):
        '''
        
        
        @param ad: The astrodata to display.
        @type ad: AstroData
        '''
        
        # do stack id and astrodata what nots.
        desc = Descriptors.getCalculator( ad )
        displayfunc = desc.fetchValue( 'display', ad )
        
        if displayfunc is None:
            displayfunc = self.ds9.displayFile
            
        if id is None:
            id = IDFactory.generateStackableID( ad )
        
        self.ds9.frame( id )
        framenumber = self.ds9[id]
        displayfunc( ad.filename, frame=framenumber, fl_imexam=False)


