import sys
from astrodata import Descriptors

class CalculatorInterface:
       
    # Descriptor Interfaces, add alphabetically
    def airmass(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.airmass(self, **args)
            
    def camera(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.camera(self, **args)
        
    def cwave(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.cwave(self, **args)
    cwave.units = "microns"
        
    def datasec(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.datasec(self, **args)
        
    def detsec(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.detsec(self, **args)
        
    def disperser(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.disperser(self, **args)
        
    def exptime(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.exptime(self, **args)
        
    def filterid(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.filterid(self, **args)
        
    
    def filtername(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.filtername(self, **args)
    filtername.units = "string"
        
    def fpmask(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.fpmask(self, **args)

    def gain(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.gain(self, **args)

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
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.nonlinear(self, **args)

    def nsciext(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.nsciext(self, **args)

    def object(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.object(self, **args)

    def obsmode(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.obsmode(self, **args)
    
    def obsepoch(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.obsepoch(self, **args)
    

    def pixscale(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.pixscale(self, **args)

    def pupilmask(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.pupilmask(self, **args)

    def rdnoise(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.rdnoise(self, **args)

    def satlevel(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.satlevel(self, **args)

    def utdate(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.utdate(self, **args)

    def uttime(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.uttime(self, **args)

    def wdelta(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.wdelta(self, **args)

    def wrefpix(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.wrefpix(self, **args)

    def xccdbin(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.xccdbin(self, **args)

    def yccdbin(self, **args):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return None # "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.yccdbin(self, **args)

    # 
    def _lazyloadCalculator(self, **args):
        """Function to put at top of all descriptor members
        to ensure the descriptor is loaded.  This way we avoid
        loading it if it is not needed."""
        if self.descriptorCalculator == None:
            self.descriptorCalculator = Descriptors.getCalculator(self, **args)
