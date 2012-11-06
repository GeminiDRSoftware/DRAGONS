from astrodata import RecipeManager
from astrodata.RecipeManager import RecipeLibrary, centralPrimitivesCatalog
import inspect
from astrodata.mkcalciface import get_calculator_interface


def get_primitives(primsetInstance):
    def pred(tob):
        return inspect.isgeneratorfunction(tob)
    membs = inspect.getmembers(primsetInstance, pred )
    names = [ a[0] for a in membs]
    # print "adi10:", repr(dir(membs[0]))
    return names

def get_descriptors(classname = None, astrotype = None):
    from astrodata.Descriptors import centralCalculatorIndex
        
    classname = centralCalculatorIndex[astrotype]
    cnp = classname.split(".")
    exec("import "+cnp[0])    
    calcobj = eval(classname)
    membs = inspect.getmembers(calcobj,inspect.ismethod)
    ret = {}
    for memb in membs:
        newdict = {}
        print "26",memb[0],repr(type(memb[1]))
        if memb[0][0] == "_":
            continue
        ret.update({memb[0]:newdict})
        newdict.update({"name":memb[0],
                        "lnum":inspect.getsourcelines(memb[1])[1],
                        "path":inspect.getsourcefile(memb[1])
                        })
          
    return {"astrotype":astrotype,
             "descriptorDict":ret}