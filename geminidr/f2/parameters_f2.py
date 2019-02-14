# This parameter file contains the parameters related to the primitives located
# in the primitives_f2.py file, in alphabetical order.
from geminidr.core import parameters_photometry


class addReferenceCatalogConfig(parameters_photometry.addReferenceCatalogConfig):
    def setDefaults(self):
        self.radius = 0.033
        self.source = "2mass"
