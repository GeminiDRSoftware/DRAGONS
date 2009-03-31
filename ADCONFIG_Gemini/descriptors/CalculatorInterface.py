import astrodata.Descriptors

class CalculatorInterface:
       
    # Descriptor Interfaces, add alphabetically
    def filtername(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.filtername(self)
        
    def gain(self):
        self.lazyloadCalculator()
        return self.descriptorCalculator.gain(self)

    # 
    def lazyloadCalculator(self):
        """Function to put at top of all descriptore members
        to ensure the descriptor is loaded.  This way we avoid
        loading it if it is not needed."""
        if self.descriptorCalculator == None:
            self.descriptorCalculator = Descriptors.getCalculator(self)
