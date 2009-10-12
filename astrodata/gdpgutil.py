import AstroDataType
import sys
from AstroData import AstroData


class GDPGUtilExcept:
    """ This is the general exception the classes and functions in the
    Structures.py module raise.
    """
    def __init__(self, msg="Exception Raised in Recipe System"):
        """This constructor takes a message to print to the user."""
        self.message = msg
    def __str__(self):
        """This str conversion member returns the message given by the user (or the default message)
        when the exception is not caught."""
        return self.message
        
def openIfName(dataset):
    """Utility function to handle accepting datasets as AstroData
    instances or string filenames. Works in conjunction with closeIfName.
    The way it works, openIfName opens returns an GeminiData isntance"""
    
    bNeedsClosing = False
    
    if type(dataset) == str:
        bNeedsClosing = True
        gd = AstroData(dataset)
    elif isinstance(dataset, AstroData):
        bNeedsClosing = False
        gd = dataset
    else:
        raise RecipeExcept("BadArgument in recipe utility function: openIfName(..)\n MUST be filename (string) or GeminiData instrument")
    
    return (gd, bNeedsClosing)
    
    
def closeIfName(dataset, bNeedsClosing):
    """Utility function to handle accepting datasets as AstroData
    instances or string filenames. Works in conjunction with openIfName."""

    if bNeedsClosing == True:
        dataset.close()
    
    return

def inheritConfig(typ, index, cl = None):
    # print "GU34:", typ, str(index)
    if cl == None:
        cl = AstroDataType.getClassificationLibrary()

    if typ in index:
        return {typ:index[typ]}
    else:
        typo = cl.getTypeObj(typ)
        supertypos = typo.getSuperTypes(oneGeneration = True)
        cfgs = {}
        for supertypo in supertypos:
            cfg = inheritConfig(supertypo.name, index, cl = cl)
            if cfg != None:
                cfgs.update({supertypo.name: cfg})
        if len(cfgs) == 0:
            return None
        else:
            return cfgs

def pickConfig(dataset, index, style = "unique"):
    """Pick config will pick the appropriate config for the style.
    NOTE: currently style must be unique, a unique config is chosen using
    inheritance.
    """
    ad,obn = openIfName(dataset)
    cl = ad.getClassificationLibrary()
    candidates = {}
    if style == "unique":
        types = ad.getTypes(prune=True)
        # print "\nGU58:", types, "\nindex:",index, "\n"
        for typ in types:
            if typ in index:
                candidates.update({typ:index[typ]})
            else:
                cfgd = inheritConfig(typ, index, cl = cl)
                if cfgd != None:
                    candidates.update(cfgd)
        #    print "\nGU61: candidates:", candidates, "\n"
            # sys.exit(1)
        # style unique this can only be one thing
        k = candidates.keys()
        if len(k)>1:
            raise (GDPGUtil)
        
    closeIfName(ad, obn)
    return candidates
