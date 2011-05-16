from astrodata.adutils.future import pyDisplay
import astrodata
from astrodata.AstroData import AstroData
from astrodata import Descriptors
from astrodata import IDFactory
from pyraf import iraf

def supported():
    return pyDisplay.supported()

class DisplayServiceException:
    """ This is the general exception the classes and functions in the
    Structures.py module raise.
    """
    def __init__(self, msg="Exception Raised in Display Service"):
        """This constructor takes a message to print to the user."""
        self.message = msg
    def __str__(self):
        """This str conversion member returns the message given by the user (or the default message)
        when the exception is not caught."""
        return self.message

_displayObj = None

def getDisplayService():
    global _displayObj
    
    if _displayObj is None:
        _displayObj = DisplayService()
    
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
        self.pydisp = pyDisplay.getDisplay()
        self.ds9 = None
        self.displayIDMap = {}
    def displayID2frame(self, displayID):
        if displayID in self.displayIDMap:
            return self.displayIDMap[displayID]
        else:
            
            for i in range(1,100):
                if not self.frameAssigned(i):
                    self.displayIDMap.update({displayID:i})
                    return i 
                
    def frameAssigned(self,i):
        dim = self.displayIDMap
        for key in dim.keys():
            if dim[key] == i:
                return True
        return False
                 
#------------------------------------------------------------------------------ 
    def setupDS9(self):
        storeDs9 = self.pydisp.getDisplayTool( 'ds9' )
        if storeDs9.pyds9_sup:
            storeDs9.set( 'tile yes' )
            storeDs9.zoomto( (1./4.) )
        self.ds9 = storeDs9
        return storeDs9
#------------------------------------------------------------------------------ 
    def getDisplayID(self, ad):
        '''
        
        
        '''
        #@@TODO:
        #=======================================================================
        # Here or IDFactory is where I believe the majority of ADType / Stacking 
        # intelligence will be done. Because ids can be passed in, the intelligence 
        # can also be done externally (in reduce, for example).
        #=======================================================================
        if 'avgcomb_' in ad.filename:
            fid = 'STACKED_' + IDFactory.generate_stackable_id( ad )
        else:
            fid = IDFactory.generate_stackable_id( ad )

        return fid
#------------------------------------------------------------------------------ 
    def display(self, ad, fid=None):
        '''
        
        
        @param ad: The astrodata to display.
        @type ad: AstroData
        '''
        
        # do stack id and astrodata what nots.
#        print 'GDS 66:'
        if type(ad) is not astrodata.AstroData:
            ad = AstroData( ad )
        desc = Descriptors.get_calculator( ad )
        
        displayfunc = desc.fetch_value( 'display', ad )
        
        if displayfunc is None:
            # This occurs when there is specific display function for the given tool.
            # (i.e. it is using the default display descriptor).
            displayfunc = self.ds9.displayFile
        
        if fid is None:
            fid = self.getDisplayID( ad )
        
        self.ds9.frame( fid )
        framenumber = self.ds9[fid]
        
        if self.ds9.pyds9_sup:
            self.ds9.set( 'regions delete all' )
        
        displayfunc( ad.filename, frame=framenumber, fl_imexam=False,
                     Stdout = coi.get_iraf_stdout(), Stderr = coi.get_iraf_stderr())
#------------------------------------------------------------------------------ 
    def markText(self, xcoord, ycoord, text='', ad=None, fid=None):
        '''
        
        '''
        line_size = 50
        
        # frame stuff
        if fid is not None:
            if ad is not None:
                self.frame( self.getDisplayID( ad ) )
            elif self.ds9.frame() != self.ds9[fid]:
                self.frame( fid )
        
        self.ds9.drawText( xcoord, ycoord, text=text, color='red')
