#!/usr/bin/env python
#
# Author: K.dement ( kdement@gemini.edu )
# Date: 5 Feb 2010
#
# Last Modified 9 Feb 2010 by kdement

from optparse import OptionParser
import os, sys
# from RECIPES_Gemini.primitives import primitives_GEMINI, primitives_GMOS_IMAGE, primitives_GMOS_OBJECT_RAW
# from primitives_GEMINI import GEMINIPrimitives
# from primitives_GMOS_IMAGE import GMOS_IMAGEPrimitives
# from primitives_GMOS_OBJECT_RAW import GMOS_OBJECT_RAWPrimitives
from sets import Set 
from copy import copy
from astrodata.AstroData import AstroData
print "about to import RecipeManager"
from astrodata import RecipeManager 
print "imported RecipeManager"
from astrodata.RecipeManager import RecipeLibrary

from utils import terminal
term = terminal
from utils.terminal import TerminalController
REASLSTDOUT = sys.stdout
REALSTDERR = sys.stderr
fstdout = terminal.FilteredStdout()
colorFilter = terminal.ColorFilter(False)
fstdout.addFilter(colorFilter)
sys.stdout = fstdout   
SW = 60

print "about to create RecipeLibrary"
rl = RecipeLibrary()
print "created RecipeLibrary"

#FROM COMMANDLINE WHEN READY
parser = OptionParser()
(options, args) = parser.parse_args()
options.args = args
options.useColor = True
options.showParams = True

options.datasets = []
options.astrotypes = []
for arg in options.args:
    if os.path.exists(arg):
        options.datasets.append(arg)
    else:
        options.astrotypes.append(arg)
    
# COLOR ON OR OFF, on by default
if options.useColor == True:
    colorFilter.on = True


#
primtypes = RecipeManager.centralPrimitivesIndex.keys()
module_list = []

path = './primitives_List.txt'
    
fhandler = open( path , 'w' )

def show(arg):
    global fhandler
    print arg
    fhandler.write(arg+"\n")

print "Create Module List"
#### ADDING MODULES TO THE MODULE_LIST
if True:
    # print "lP62:",repr(options.astrotypes), repr(options.datasets)
    if len(options.astrotypes) or len(options.datasets):
        for typ in options.astrotypes:
            ps = rl.retrievePrimitiveSet(astrotype = typ)        
            if ps != None:
                module_list.append(ps)
        for dataset in options.datasets:
            ad = AstroData(dataset)
            ps = rl.retrievePrimitiveSet(dataset = ad)
            s = "%(ds)s-->%(typ)s" % {"ds": dataset,
                                                        "typ": ps.astrotype}
            p = " "*(SW - len(s))
            show("${REVERSE}"+s+p+"${NORMAL}")
            if ps:
                module_list.append(ps)
    else:
        for key in primtypes:
            module_list.append(rl.retrievePrimitiveSet(astrotype = key))   
print "Done Creating Module List"

if len(module_list) == 0:
    print "Found no primitive sets associated with:"
    for arg in options.args:
        print "   ",arg

geminiList=[]
childList=[]
intersectionList=[]
outerloop = 0
primsdict = {}
name2class = {}
primsdictKBN = {}
class2instance = {}

def getPrimList(cl):
    plist = []
    for key in cl.__dict__:
        doappend = True
        fob = eval("cl."+key)
        if not hasattr(fob, "__call__"):
            doappend = False
        
        if hasattr(fob, "pt_hide"):
            doappend = eval ("not fob.pt_hide")
            
        if key.startswith("_"):
            doappend = False   
        if doappend:
            plist.append( key )
    plist.sort()
    return plist
    
def constructPrimsDict(primset, primsdict=None):
    primsdict.update({primset:getPrimList(primset.__class__)})
    
def constructPrimsclassDict(startclass):
    if startclass.__name__== "PrimitiveSet":
        return
    global primsdict, primsdictKBN, name2class
    name2class.update({startclass.__name__:startclass})
    primsdictKBN.update({startclass.__name__:getPrimList(startclass)})
    for base in startclass.__bases__:
        constructPrimsclassDict(base)

primsdict = {}
for primset in module_list:
    pname = primset.__class__.__name__
    print "constructing Primitive Dictionary for "+pname
    constructPrimsDict(primset, primsdict)
    print "construction Primitive Class Dictionary for "+pname
    constructPrimsclassDict(primset.__class__)
    class2instance.update({pname:primset})

# get a sorted list of primitive sets, sorted with parents first
def primsetcmp(a,b):
    if isinstance(a,type(b)):
        return 1
    elif isinstance(b,type(a)):
        return -1
    else:
        an = a.__class__.__name__
        bn = b.__class__.__name__
        print an, bn, an>bn
        if an > bn:
            return 1
        elif an < bn:
            return -1
        else:
            return 0
                    
primsets = primsdict.keys()
primsets.sort(primsetcmp)

def firstprim(primsetname, prim):
    global name2class, primsdictKBN
    if primsetname in primsdictKBN:
        if prim in primsdictKBN[primsetname]:
            return primsetname
        else:
            cl = name2class[primsetname]
            for base in cl.__bases__:
                fp = firstprim(base.__name__, prim)
                if fp:
                    return fp
            return None
    else:
        return None
 
def overrides(primsetname, prim):
    global name2class, primsdictKBN
    
    cl = name2class[primsetname]
    for base in cl.__bases__:
        fp = firstprim(base.__name__, prim)
        return fp
            
    return None
    
def showPrims(primsetname, primset=None, i = 0, indent = 0, pdat = None):
    INDENT = " "
    indentstr = INDENT*indent
    if primset == None:
        firstset = True
    else:
        firstset = False
        
    if primset == None:
        primlist = primsdictKBN[primsetname]
        primset = copy(primlist)
    else:
        myprimset = Set(primsdictKBN[primsetname])
        givenprimset = Set(primset)
        
        prims = myprimset - givenprimset
        
        primlist = list(prims)
        primlist.sort()
        primset.extend(primlist)
        
    cl = name2class[primsetname]
    if firstset:
        show("${BOLD}"+'_'*SW+"${NORMAL}")
        show("\n${BOLD}%s${NORMAL} Primitive Set (class: %s)" % (cl.astrotype,primsetname))
        show("-"*SW)
    else:
        if len(primlist)>0:
            show("${BLUE}%s(Following Are Inherited from %s)${NORMAL}" % (INDENT*indent, primsetname))
        
    
    for prim in primlist:
        i+=1
        over = overrides(primsetname, prim)
        primline = "%s%2d. %s" % (" "*indent, i, prim)
        if over:
            primline += "  ${BLUE}(overrides %s)${NORMAL}" % over
        show(primline)
        if options.showParams:
                indent0 = indentstr+INDENT*5
                indent1 = indentstr+INDENT*6
                
                indentp = indent1+"parameter: "
                indentm = indent1+INDENT*2
                primsetinst = class2instance[primsetname]
                if pdat == None:
                    paramdicttype = primsetinst.astrotype
                    paramdict = primsetinst.paramDict
                    
                    pdat = (paramdicttype,paramdict)
                else:
                    paramdicttype = pdat[0]
                    paramdict = pdat[1]

                for primname in paramdict.keys():
                    if primname == prim:
                        if not firstset:
                            show( term.GREEN
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
                            show(term.GREEN+indentp+term.NORMAL+paramname+term.NORMAL)
                            metadata = paramdict[primname][paramname].keys()
                            maxlen = len(max(metadata, key=len)) + 3
                            metadata.sort()
                            if "default" in metadata:
                                metadata.remove("default")
                                metadata.insert(0,"default")
                            for metadatum in metadata:
                                padding = " "*(maxlen - len(metadatum))
                                val = paramdict[primname][paramname][metadatum]
                                show(term.GREEN+indentm+metadatum+padding+"= "+repr(val)+term.NORMAL)
        
                
    for base in cl.__bases__:
        if base.__name__ in primsdictKBN:
            showPrims(base.__name__, primset = primset, i = i, indent = indent+2, pdat = pdat)        


for primset in primsdict:
    showPrims(primset.__class__.__name__)
    show("\n")

fhandler.close()
