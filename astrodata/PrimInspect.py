#!/usr/bin/env python


import os, sys
import re
#get color printing started
from astrodata.adutils import terminal
term = terminal
from astrodata.adutils.terminal import TerminalController
REASLSTDOUT = sys.stdout
REALSTDERR = sys.stderr
fstdout = terminal.FilteredStdout()
colorFilter = terminal.ColorFilter(True)
fstdout.addFilter(colorFilter)
sys.stdout = fstdout   
SW = 80

#from optparse import OptionParser
from sets import Set 
from copy import copy
from astrodata.AstroData import AstroData
from astrodata import RecipeManager 
from astrodata.RecipeManager import RecipeLibrary
from inspect import getsourcefile

rl = RecipeLibrary()
colorFilter.on = False
primtypes = RecipeManager.centralPrimitivesIndex.keys()

class PrimInspect():
    '''
    object for use with listPrimitives.py
    '''
    module_list = None
    path = "./primitives_List.txt"
    fhandler = None
    datasets = None
    astrotypes = None
    primsdict = None
    name2class = None
    primsdictKBN = None
    class2instance = None
    options = None
    
    def __init__(self, options=None, path = None):
        self.module_list = []
        if path:
            self.path = path
        self.module_list = []
        self.datasets = []
        self.astrotypes = []
        self.primsdict = {}
        self.name2class = {}
        self.primsdictKBN = {}
        self.class2instance = {}
        self.options = options
        if self.options.verbose:
            self.options.useColor = True
            self.options.showParams = True
            self.options.showUsage = True
            self.options.showInfo = True
        if self.options.useColor:
            colorFilter.on = True
        if self.options.makeOutputFile:
            self.fhandler = open( self.path , 'w' )
            
            
        #----------------------------------------------       
        
    def show(self, arg):
        print arg
        if self.options.makeOutputFile:
            arg = re.sub(r'\$\$|\${\w+}','',arg)
            # replaced by above
            # arg = arg.replace('${BOLD}','')
            # arg = arg.replace('${NORMAL}','')
            # arg = arg.replace('${YELLOW}','')
            # arg = arg.replace('${RED}','')
            # arg = arg.replace('${GREEN}','')
            # arg = arg.replace('${BLUE}','')
            self.fhandler.write(arg+"\n")
        
    def close_fhandler(self):
        if self.options.makeOutputFile:
            self.fhandler.close()
        
    def createModule_list(self):
        if len(self.astrotypes) or len(self.datasets):
            if self.astrotypes:
                self.astrotypes.sort()
            for typ in self.astrotypes:
                ps = rl.retrievePrimitiveSet(astrotype = typ)        
                if ps != None:
                    self.module_list.extend(ps)
            for dataset in self.datasets:
                ad = AstroData(dataset)
                ps = rl.retrievePrimitiveSet(dataset = ad)
                s = "%(ds)s-->%(typ)s" % {"ds": dataset, "typ": ps[0].astrotype}
                p = " "*(SW - len(s))
                self.show("${YELLOW}"+s+p+"${NORMAL}")
                if ps:
                    self.module_list.extend(ps)
        else:
            for key in primtypes:
                try:
                    self.module_list.extend(rl.retrievePrimitiveSet(astrotype = key))
                except:
                    self.show("${RED}ERROR: cannot load primitive set for astrotype %s${NORMAL}"
                                    % key)
        if len(self.module_list) == 0:
            print "Found no primitive sets associated with:"
            for arg in self.options.args:
                print "   ",arg
    
    # get a sorted list of primitive sets, sorted with parents first

            
    def getPrimList( self, cl ):
        plist = []
        for key in cl.__dict__:
            doappend = True
            fob = eval( "cl."+key )
            if not hasattr( fob, "__call__" ):
                doappend = False
            if hasattr( fob, "pt_hide" ):
                doappend = eval ( "not fob.pt_hide" )
            elif hasattr( cl, "pthide_"+key ):
                doappend = eval ( "not cl.pthide_"+key )
                
            if key.startswith( "_" ):
                doappend = False   
            if doappend:
                plist.append( key )
        plist.sort()
        return plist
    
    def constructPrimsDict( self, primset ):
        self.primsdict.update( { primset:self.getPrimList( primset.__class__ ) } )
    
    def constructPrimsclassDict( self, startclass ):
        if startclass.__name__== "PrimitiveSet":
            return
        self.name2class.update( {startclass.__name__:startclass} )
        self.primsdictKBN.update( { startclass.__name__:self.getPrimList( startclass ) } )
        for base in startclass.__bases__:
            self.constructPrimsclassDict( base )
    
    def buildDictionaries(self):
        self.createModule_list()
        for primset in self.module_list:
            pname = primset.__class__.__name__
            self.constructPrimsDict(primset)
            self.constructPrimsclassDict(primset.__class__)
            self.class2instance.update({pname:primset})
      

    def firstprim(self, primsetname, prim):
        if primsetname in self.primsdictKBN:
            if prim in self.primsdictKBN[primsetname]:
                return primsetname
            else:
                cl = self.name2class[primsetname]
                for base in cl.__bases__:
                    fp = self.firstprim(base.__name__, prim)
                    if fp:
                        return fp
                return None
        else:
            return None
        
    def hides(self, primsetname, prim, instance=None):
        """checks to see if prim hides or is hidden-by another"""
        
        if instance:
            cl = self.name2class[primsetname]
            ps = instance
            psl = rl.retrievePrimitiveSet(astrotype= instance.astrotype)
        elif primsetname in self.class2instance:
            ps = self.class2instance[primsetname]
            cl = ps.__class__
            psl = rl.retrievePrimitiveSet(astrotype= ps.astrotype)
        else:
            return None
        verb = False # prim == "exit"
        if verb: print "lP200:", len(psl)
        if len(psl)>1:
            before = True
            rets = None
            for ops in psl:
                
                # reason for this comparison: make this work even
                # if retrievePrimitiveSet returns new instances...
                if ps.__class__.__name__ == ops.__class__.__name__:
                    before = False
                    if verb : print "lP209: found this by by class name", repr(ops)
                    continue
                if isinstance(ops, cl):
                    before = False
                    if verb : print "lp213: skipping due to isinstance"
                    continue
                if verb: print "lP215:", repr(ops),repr(dir(ops))
                if hasattr(ops, prim):
                    if verb : print "lP216: hide happens"
                    if before:
                        rets = "${RED}(hidden by "+ops.__class__.__name__+")${NORMAL}"
                    else:
                        rets = '${GREEN}(hides "%s" from %s)${NORMAL}' %(prim, ops.__class__.__name__)
                    break
            return rets
        return None

    def primsetcmp(self,a,b):
        if isinstance(a,type(b)):
            return 1
        elif isinstance(b,type(a)):
            return -1
    
        else:
            return 0
    
    def overrides(self, primsetname, prim):
        cl = self.name2class[primsetname]
        for base in cl.__bases__:
            fp = self.firstprim(base.__name__, prim)
            return fp
        return None
    
    def showSetInfo(self, primsetname, cl, primlist):        
        sfull = getsourcefile(cl)
        sdir = os.path.dirname(sfull)
        sfil = os.path.basename(sfull)
        self.show("  Class            : ${BOLD}"+primsetname+"${NORMAL}")
        self.show("  Description      : ${BOLD}"+cl.astrotype+" Primitive Set${NORMAL}")
        inherit = 'None'
        inherit_list=[]
        for base in cl.__bases__:
            if base.__name__ in self.primsdictKBN:
                inherit_list.append( base.__name__ )
                inherit = 'Yes'
        self.show("  Inheritance      : ${BOLD}"+inherit+"${NORMAL}")
        if inherit is 'Yes':
            #right now only works with one level of inheritence
            overrides_count=0
            for prim in primlist:
                over = self.overrides(primsetname, prim)
                if over:
                    overrides_count+=1
            itot = 0
            for inherited in inherit_list:
                self.show("                   : (from ${BOLD}"+inherited+"${NORMAL}")
                if len(inherit_list) < 2:
                    iprimlist = self.primsdictKBN[inherited]
                    itot = itot + (len(iprimlist) - overrides_count)
                    len_iprimlist = str( len(iprimlist) - overrides_count )
                    self.show("                   : inherited ${BOLD}"+len_iprimlist+"${NORMAL} primitives")
                    self.show("                   : with ${BOLD}"+str(overrides_count)+" overridden)${NORMAL} ")
        len_primlist = str( len(primlist) )
        self.show("  Local primitives :${BOLD} "+len_primlist+"${NORMAL}")
        if len( inherit_list ) is 1:
            self.show("  Total primitives : ${BOLD}"+str(itot + len(primlist))+"${NORMAL}")
        self.show("  Source File      : ${BOLD}"+sfil+"${NORMAL}")
        self.show("  Location         : ${BOLD}"+sdir+"${NORMAL}")

    def showPrims(  self,primsetname, 
                    primset=None, 
                    i = 0, 
                    indent = 0, 
                    pdat = None,
                    instance = None):
        INDENT = " "
        indentstr = INDENT*indent
        if primset == None:
            firstset = True
        else:
            firstset = False
            
        if firstset == True:
            primlist = self.primsdictKBN[primsetname]
            primset = copy(primlist)
        else:
            myprimset = Set(self.primsdictKBN[primsetname])
            givenprimset = Set(primset)
            
            prims = myprimset - givenprimset
            
            primlist = list(prims)
            primlist.sort()
            primset.extend(primlist)
        cl = self.name2class[primsetname]
       
        if firstset:           
            self.show("${BOLD}"+'_'*SW+"${NORMAL}")
            if self.options.showInfo:
                self.show("\n${BOLD}%s${NORMAL}\n" % (cl.astrotype))
                self.showSetInfo(primsetname, cl, primlist) 
            else:
                self.show("\n${BOLD}%s ${NORMAL}(%s)\n" % (cl.astrotype, primsetname))
            self.show("-"*SW)
            astrotype = cl.astrotype
            instance = self.class2instance[primsetname]
        else:
            if len(primlist)>0:
                self.show("${BLUE}%s(Following Are Inherited from %s)${NORMAL}" % (INDENT*indent, primsetname))
        
        if len(primlist) == 0:
            maxlenprim = 0
        else:
            maxlenprim = min(16, len(max(primlist, key=len)))
        for prim in primlist:
            i+=1
            hide = self.hides(primsetname, prim, instance = instance)
            over = self.overrides(primsetname, prim)
            primline = "%s%2d. %s" % (" "*indent, i, prim)
            pl = len(prim)
            if pl < maxlenprim:
                primline += " "*(maxlenprim-pl)
            if over:
                primline += "  ${BLUE}(overrides %s)${NORMAL}" % over
            if hide:
                primline += "  %s" % hide
            self.show(primline)
            if self.options.showUsage:
                func = eval("instance."+prim)
                if hasattr(func, "pt_usage"):
                    print " "*indent+'    ${YELLOW}DOC:'+eval("func.pt_usage")+'${NORMAL}'
                if hasattr(instance, "ptusage_"+prim):
                    print " "*indent+'    ${YELLOW}DOC: '+eval("instance.ptusage_"+prim)+'${NORMAL}'
            if self.options.showParams:
                indent0 = indentstr+INDENT*5
                indent1 = indentstr+INDENT*6
                
                indentp = indent1+"parameter: "
                indentm = indent1+INDENT*2
                if pdat == None:
                    primsetinst = self.class2instance[primsetname]
                    paramdicttype = primsetinst.astrotype
                    paramdict = primsetinst.paramDict
                    
                    pdat = (paramdicttype,paramdict)
                else:
                    paramdicttype = pdat[0]
                    paramdict = pdat[1]
                for primname in paramdict.keys():
                    if primname == prim:
                        if not firstset:
                            self.show( term.GREEN
                                + term.BOLD
                                + indent0
                                + "(these parameter settings for an inherited primitive still originate in"
                                + "\n"
                                + indent0
                                + " the ${BLUE}%s${GREEN} Primitive Set Parameters)${NORMAL}"% paramdicttype
                                + term.NORMAL)

                        paramnames = paramdict[primname].keys()
                        paramnames.sort()
                        for paramname in paramnames:
                            self.show(term.GREEN+indentp+term.NORMAL+paramname+term.NORMAL)
                            metadata = paramdict[primname][paramname].keys()
                            maxlen = len(max(metadata, key=len)) + 3
                            metadata.sort()
                            if "default" in metadata:
                                metadata.remove("default")
                                metadata.insert(0,"default")
                            for metadatum in metadata:
                                padding = " "*(maxlen - len(metadatum))
                                val = paramdict[primname][paramname][metadatum]
                                self.show(term.GREEN+indentm+metadatum+padding+"= "+repr(val)+term.NORMAL)
        
                        
        for base in cl.__bases__:
            if base.__name__ in self.primsdictKBN:
                self.showPrims(  base.__name__,
                            primset = primset, 
                            i = i, indent = indent+2, 
                            pdat = pdat, instance = instance)        

