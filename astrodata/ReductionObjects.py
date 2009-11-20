    
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
        prim = eval("self.%s" % primname)
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
