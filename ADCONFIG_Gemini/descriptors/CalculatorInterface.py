import sys
from astrodata import Descriptors

class CalculatorInterface:
       
    # Descriptor Interfaces, add alphabetically
    def airmass(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.airmass(self)
            
    def camera(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.camera(self)
        
    def cwave(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.cwave(self)
        
    def datasec(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.datasec(self)
        
    def detsec(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.detsec(self)
        
    def disperser(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.disperser(self)
        
    def exptime(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.exptime(self)
        
    def filterid(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.filterid(self)
        
    def filtername(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.filtername(self)
        
    def fpmask(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.fpmask(self)

    def gain(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.gain(self)

    def instrument(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.instrument(self)

    def mdfrow(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.mdfrow(self)

    def nonlinear(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.nonlinear(self)

    def nsciext(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.nsciext(self)

    def object(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.object(self)

    def obsmode(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.obsmode(self)

    def pixscale(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.pixscale(self)

    def pupilmask(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.pupilmask(self)

    def rdnoise(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.rdnoise(self)

    def satlevel(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.satlevel(self)

    def utdate(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.utdate(self)

    def uttime(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.uttime(self)

    def wdelta(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.wdelta(self)

    def wrefpix(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.wrefpix(self)

    def xccdbin(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.xccdbin(self)

    def yccdbin(self):
        self._lazyloadCalculator()
        descriptorname = sys._getframe().f_code.co_name
        if not hasattr( self.descriptorCalculator, descriptorname):
            return "Base Calculator Class does not have %s function" % descriptorname
        return self.descriptorCalculator.yccdbin(self)

    # 
    def _lazyloadCalculator(self):
        """Function to put at top of all descriptor members
        to ensure the descriptor is loaded.  This way we avoid
        loading it if it is not needed."""
        if self.descriptorCalculator == None:
            self.descriptorCalculator = Descriptors.getCalculator(self)
