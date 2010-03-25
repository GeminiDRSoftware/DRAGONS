#!/usr/bin/env python
import os, sys
# from RECIPES_Gemini.primitives import primitives_GEMINI, primitives_GMOS_IMAGE, primitives_GMOS_OBJECT_RAW
# from primitives_GEMINI import GEMINIPrimitives
# from primitives_GMOS_IMAGE import GMOS_IMAGEPrimitives
# from primitives_GMOS_OBJECT_RAW import GMOS_OBJECT_RAWPrimitives
from sets import Set 
from copy import copy
from astrodata import RecipeManager 
from astrodata.RecipeManager import RecipeLibrary
rl = RecipeLibrary()
SW = 60

# Description: 'listPrimitives' is a simple script to list available primitives both to screen and to a file
#      ( located in RECIPES_Gemini/primitives folder, primitives_List.txt ). In addition, when there are primitives  
#      with the exact same names as those in  'GEMINIPrimitives', then a list of these will be provided with a proper
#      heading at the bottom of the associated primitive list.  This allows users to quickly find out what primitives
#      override GEMINI. 
#
#      * To keep script up-to-date, one must update imports above and module_list below when adding, removing or 
#        renaming new primitive classes.
#
# Author: K.dement ( kdement@gemini.edu )
# Date: 5 Feb 2010
#
# Last Modified 9 Feb 2010 by kdement

primtypes = RecipeManager.centralPrimitivesIndex.keys()
module_list = []

path = './primitives_List.txt'
    
fhandler = open( path , 'w' )

def show(arg):
    global fhandler
    print arg
    fhandler.write(arg+"\n")

for key in primtypes:
    module_list.append(rl.retrievePrimitiveSet(astrotype = key))   

geminiList=[]
childList=[]
intersectionList=[]
outerloop = 0
primsdict = {}
name2class = {}
primsdictKBN = {}

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
    constructPrimsDict(primset, primsdict)
    constructPrimsclassDict(primset.__class__)

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
    
def showPrims(primsetname, primset=None, i = 0, indent = 0):
    
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
        
    
    if firstset:
        show('_'*SW)
        show("\n"+primsetname)
        show("-"*SW)
    else:
        if len(primlist)>0:
            show("%s(Inherited from %s)" % (" "*indent, primsetname))
        
    for prim in primlist:
        i+=1
        over = overrides(primsetname, prim)
        primline = "%s%2d. %s" % (" "*indent, i, prim)
        if over:
            primline += " (overrides %s)" % over
        show(primline)
     
    cl = name2class[primsetname]
    for base in cl.__bases__:
        if base.__name__ in primsdictKBN:
            showPrims(base.__name__, primset = primset, i = i, indent = indent+2)        


for primset in primsdict:
    showPrims(primset.__class__.__name__)
    show("\n")
        
show( '\n\n' )
fhandler.close()
