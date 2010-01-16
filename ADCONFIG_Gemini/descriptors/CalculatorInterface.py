import sys
from astrodata import Descriptors

class CalculatorInterface:
       
    # Descriptor Interfaces, add alphabetically
    def airmass(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.airmass(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
            
    def camera(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.camera(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
            
    def cwave(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.cwave(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
            
        
    def datasec(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.datasec(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
            
    def detsec(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.detsec(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
            
    def disperser(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.disperser(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
            
    def exptime(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.exptime(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
            
    def filterid(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.filterid(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
            
    
    def filtername(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.filtername(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
            
        
    def fpmask(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.fpmask(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def gain(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.gain(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
            
    def instrument(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.instrument(self, **args)

    def mdfrow(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.mdfrow(self, **args)

    def nonlinear(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.nonlinear(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
            
    def nsciext(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.nsciext(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
            
    def object(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.object(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
                
    def obsmode(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.obsmode(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
            
    def obsepoch(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.obsepoch(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
            
    def pixscale(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.pixscale(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
                
    def pupilmask(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.pupilmask(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
            
    def rdnoise(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.rdnoise(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
            
    def satlevel(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.satlevel(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def utdate(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.utdate(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
            
    def uttime(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.uttime(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None

    def wdelta(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.wdelta(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
            
    def wrefpix(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.wrefpix(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
            
                
    def xccdbin(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.xccdbin(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
            
    def yccdbin(self, **args):
        try:
            self._lazyloadCalculator()
            descriptorname = sys._getframe().f_code.co_name
            if not hasattr( self.descriptorCalculator, descriptorname):
                return None # "Base Calculator Class does not have %s function" % descriptorname
            return self.descriptorCalculator.yccdbin(self, **args)
        except:
            self.noneMsg = str(sys.exc_info()[1])
            return None
# UTILITY FUNCTIONS, above are descriptor thunks            
    def _lazyloadCalculator(self, **args):
        """Function to put at top of all descriptor members
        to ensure the descriptor is loaded.  This way we avoid
        loading it if it is not needed."""
        if self.descriptorCalculator == None:
            self.descriptorCalculator = Descriptors.getCalculator(self, **args)
