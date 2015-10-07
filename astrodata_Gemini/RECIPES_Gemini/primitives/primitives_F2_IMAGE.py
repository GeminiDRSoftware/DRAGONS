import numpy as np

from astrodata.utils import logutils
from gempy.gemini    import gemini_tools as gt

from primitives_F2 import F2Primitives


class F2_IMAGEPrimitives(F2Primitives):
    """
    This is the class containing all of the primitives for the F2_IMAGE
    level of the type hierarchy tree. It inherits all the primitives from the
    level above, 'F2Primitives'.
    """
    astrotype = "F2_IMAGE"
    
    def init(self, rc):
        F2Primitives.init(self, rc)
        return rc
    

    def selectFlatRecipe(self, rc):

        adinput = rc.get_inputs_as_astrodata()        
        recipe_list = []
        
        if adinput[0].wavelength_band().as_pytype() == 'K':
            recipe_list.append("makeLampOffFlat")
        else:
            recipe_list.append("makeLampOnLampOffFlat")
            
        rc.run("\n".join(recipe_list))
        
        yield rc
