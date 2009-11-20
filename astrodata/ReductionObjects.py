class ReductionExcept:
    """ This is the general exception the classes and functions in the
    Structures.py module raise.
    """
    def __init__(self, msg="Exception Raised by ReductionObject"):
        """This constructor takes a message to print to the user."""
        self.message = msg
    def __str__(self):
        """This str conversion member returns the message given by the user (or the default message)
        when the exception is not caught."""
        return self.message
        
class ReductionObject(object):

    recipeLib = None
    context = None
    
    def init(self, co):
        """ This member is purely for overwriting.  Controllers should call this
        before iterating over the steps of the recipe"""
        self.context = co
        return co
    
    def substeps(self, primname, context):
        self.recipeLib.checkAndBind(self, primname, context=context) 
        # print "substeps(%s,%s)" % (primname, str(cfgobj))
        if hasattr(self, primname):
            prim = eval("self.%s" % primname)
        else:
            msg = "There is no recipe or primitive named \"%s\" in ReductionObject %s" % (primname, str(repr(self)))
            raise ReductionExcept(msg)
                
        context.begin(primname)
        try:
            for co in prim(context):
                yield co
        except:
            print "%(name)s failed due to an exception." %{'name':primname}
            raise
        context.curPrimName = None
        yield context.end(primname)
        
    def runstep(self, primname, cfgobj):
        cfgobj.status = "RUNNING"
        for cfg in self.substeps(primname, cfgobj):
            pass
        return cfg
