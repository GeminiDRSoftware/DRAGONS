
import sys
from astrodata import Descriptors

class CalculatorInterface:

    def airmass(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "airmass"):
                return None 
            return self.descriptorCalculator.airmass(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def camera(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "camera"):
                return None 
            return self.descriptorCalculator.camera(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def cwave(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "cwave"):
                return None 
            return self.descriptorCalculator.cwave(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def datasec(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "datasec"):
                return None 
            return self.descriptorCalculator.datasec(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def detsec(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "detsec"):
                return None 
            return self.descriptorCalculator.detsec(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def disperser(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "disperser"):
                return None 
            return self.descriptorCalculator.disperser(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def exptime(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "exptime"):
                return None 
            return self.descriptorCalculator.exptime(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def filterid(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "filterid"):
                return None 
            return self.descriptorCalculator.filterid(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def filtername(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "filtername"):
                return None 
            return self.descriptorCalculator.filtername(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def fpmask(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "fpmask"):
                return None 
            return self.descriptorCalculator.fpmask(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def gain(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "gain"):
                return None 
            return self.descriptorCalculator.gain(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def instrument(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "instrument"):
                return None 
            return self.descriptorCalculator.instrument(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def mdfrow(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "mdfrow"):
                return None 
            return self.descriptorCalculator.mdfrow(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def nonlinear(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "nonlinear"):
                return None 
            return self.descriptorCalculator.nonlinear(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def nsciext(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "nsciext"):
                return None 
            return self.descriptorCalculator.nsciext(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def object(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "object"):
                return None 
            return self.descriptorCalculator.object(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def obsmode(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "obsmode"):
                return None 
            return self.descriptorCalculator.obsmode(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def obsepoch(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "obsepoch"):
                return None 
            return self.descriptorCalculator.obsepoch(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def pixscale(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "pixscale"):
                return None 
            return self.descriptorCalculator.pixscale(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def pupilmask(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "pupilmask"):
                return None 
            return self.descriptorCalculator.pupilmask(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def rdnoise(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "rdnoise"):
                return None 
            return self.descriptorCalculator.rdnoise(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def satlevel(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "satlevel"):
                return None 
            return self.descriptorCalculator.satlevel(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def utdate(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "utdate"):
                return None 
            return self.descriptorCalculator.utdate(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def uttime(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "uttime"):
                return None 
            return self.descriptorCalculator.uttime(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def wdelta(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "wdelta"):
                return None 
            return self.descriptorCalculator.wdelta(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def wrefpix(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "wrefpix"):
                return None 
            return self.descriptorCalculator.wrefpix(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def xccdbin(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "xccdbin"):
                return None 
            return self.descriptorCalculator.xccdbin(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def yccdbin(self, **args):
        try:
            self._lazyloadCalculator()
            if not hasattr( self.descriptorCalculator, "yccdbin"):
                return None 
            return self.descriptorCalculator.yccdbin(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
# UTILITY FUNCTIONS, above are descriptor thunks            
    def _lazyloadCalculator(self, **args):
        '''Function to put at top of all descriptor members
        to ensure the descriptor is loaded.  This way we avoid
        loading it if it is not needed.'''
        if self.descriptorCalculator == None:
            self.descriptorCalculator = Descriptors.getCalculator(self, **args)

