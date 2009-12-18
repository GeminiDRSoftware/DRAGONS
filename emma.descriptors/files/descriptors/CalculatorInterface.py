import Descriptors

class CalculatorInterface:
       
    # Descriptor Interfaces, add alphabetically
    def airmass(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.airmass(self)
        
    def camera(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.camera(self)
        
    def cwave(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.cwave(self)
        
    def datasec(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.datasec(self)
        
    def detsec(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.detsec(self)
        
    def disperser(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.disperser(self)
        
    def exptime(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.exptime(self)
        
    def filterid(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.filterid(self)
        
    def filtername(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.filtername(self)
        
    def fpmask(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.fpmask(self)

    def gain(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.gain(self)

    def instrument(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.instrument(self)

    def mdfrow(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.mdfrow(self)

    def nonlinear(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.nonlinear(self)

    def nsciext(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.nsciext(self)

    def object(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.object(self)

    def obsmode(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.obsmode(self)

    def pixscale(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.pixscale(self)

    def pupilmask(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.pupilmask(self)

    def rdnoise(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.rdnoise(self)

    def satlevel(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.satlevel(self)

    def utdate(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.utdate(self)

    def uttime(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.uttime(self)

    def wdelta(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.wdelta(self)

    def wrefpix(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.wrefpix(self)

    def xccdbin(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.xccdbin(self)

    def yccdbin(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.yccdbin(self)

    # 
    def lazyloadCalculator(self):
        """Function to put at top of all descriptor members
        to ensure the descriptor is loaded.  This way we avoid
        loading it if it is not needed."""
        if self.descriptorCalculator == None:
            self.descriptorCalculator = Descriptors.getCalculator(self)
