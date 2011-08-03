#!/usr/bin/env python
import os
import sys
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
SW = 79

#from optparse import OptionParser
from sets import Set 
from copy import copy
from astrodata.AstroData import AstroData
from astrodata import RecipeManager 
from astrodata.RecipeManager import RecipeLibrary
from inspect import getsourcefile
from astrodata import Errors
rl = RecipeLibrary()
colorFilter.on = False
primtypes = RecipeManager.centralPrimitivesIndex.keys()

class PrimInspect():
    """Tool for listing primitives, parameters, and recipes 
    """
    module_list = None
    path = "./primitives_list.txt"
    fhandler = None
    primsdict = None
    name2class = None
    primsdict_kbn = None
    class2instance = None
    primsets = None
    
    def __init__(self, use_color=False, show_param=False, show_usage=False,
                 show_info=False, make_file=False, verbose=False, path=None,
                 datasets=None, astrotypes=None):
        self.module_list = []
        if path:
            self.path = path
        self.module_list = []
        self.datasets = datasets
        self.astrotypes = astrotypes
        self.primsdict = {}
        self.name2class = {}
        self.primsdict_kbn = {}
        self.class2instance = {}
        self.verbose = verbose
        self.use_color = use_color
        self.show_param = show_param 
        self.show_usage = show_usage
        self.show_info = show_info
        self.make_file = make_file
        if self.verbose:
            self.use_color = True
            show_param = True
            self.show_usage = True
            self.show_info = True
        if self.use_color:
            colorFilter.on = True
        if self.make_file:
            self.fhandler = open( self.path , "w" )
        self.build_dictionaries()
        self.primsets = self.primsdict.keys()
        self.primsets.sort(self.primsetcmp)

    def show(self, arg):
        print arg
        if self.make_file:
            arg = re.sub(r"\$\$|\${\w+}","",arg)
            # replaced by re.sub above
            # arg = arg.replace("${<ATTR>}","")
            self.fhandler.write(arg+"\n")
        
    def close_fhandler(self):
        if self.make_file:
            self.fhandler.close()
        
    def create_module_list(self):
        if len(self.astrotypes) or len(self.datasets):
            badtype = []
            if self.astrotypes:
                self.astrotypes.sort()
            for typ in self.astrotypes:
                ps = rl.retrieve_primitive_set(astrotype=typ)        
                if ps != None:
                    self.module_list.extend(ps)
                else:
                    badtype.append(typ)
            for dataset in self.datasets:
                ad = AstroData(dataset)
                ps = rl.retrieve_primitive_set(dataset=ad)
                s = "%(ds)s-->%(typ)s" % \
                    {"ds": dataset, "typ": ps[0].astrotype}
                p = " "*(SW - len(s))
                self.show("${YELLOW}"+s+p+"${NORMAL}")
                if ps:
                    self.module_list.extend(ps)
                else:
                    badtype.append(dataset)
        else:
            for key in primtypes:
                try:
                    self.module_list.extend(\
                        rl.retrieve_primitive_set(astrotype=key))
                except:
                    self.show("${RED}ERROR: Cannot load primitive set for "
                              "astrotype %s${NORMAL}" % key)
        if len(self.module_list) == 0:
            mes = "Cannot find associated primitives with %s" % str(badtype)
            raise Errors.PrimInspectError(mes)
             
    # get a sorted list of primitive sets, sorted with parents first
    def get_prim_list(self, cl):
        plist = []
        for key in cl.__dict__:
            doappend = True
            fob = eval("cl." + key)
            if not hasattr(fob, "__call__"):
                doappend = False
            if hasattr(fob, "pt_hide"):
                doappend = eval("not fob.pt_hide")
            elif hasattr(cl, "pthide_" + key):
                doappend = eval("not cl.pthide_" + key)
            if key.startswith( "_" ):
                doappend = False   
            if doappend:
                plist.append(key)
        plist.sort()
        return plist
    
    def construct_prims_dict(self, primset):
        self.primsdict.update({primset:self.get_prim_list(\
            primset.__class__)})
    
    def construct_primsclass_dict(self, startclass):
        if startclass.__name__== "PrimitiveSet":
            return
        self.name2class.update({startclass.__name__:startclass})
        self.primsdict_kbn.update({startclass.__name__:self.get_prim_list(\
            startclass)})
        for base in startclass.__bases__:
            self.construct_primsclass_dict(base)
    
    def build_dictionaries(self):
        self.create_module_list()
        for primset in self.module_list:
            pname = primset.__class__.__name__
            self.construct_prims_dict(primset)
            self.construct_primsclass_dict(primset.__class__)
            self.class2instance.update({pname:primset})
      

    def firstprim(self, primsetname, prim):
        if primsetname in self.primsdict_kbn:
            if prim in self.primsdict_kbn[primsetname]:
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
            psl = rl.retrieve_primitive_set(astrotype= instance.astrotype)
        elif primsetname in self.class2instance:
            ps = self.class2instance[primsetname]
            cl = ps.__class__
            psl = rl.retrieve_primitive_set(astrotype= ps.astrotype)
        else:
            return None
        verb = False # prim == "exit"
        if verb: print "lP200:", len(psl)
        if len(psl)>1:
            before = True
            rets = None
            for ops in psl:
                
                # reason for this comparison: make this work even
                # if retrieve_primitive_set returns new instances...
                if ps.__class__.__name__ == ops.__class__.__name__:
                    before = False
                    if verb : print "lP209: found this by by class name",\
                        repr(ops)
                    continue
                if isinstance(ops, cl):
                    before = False
                    if verb : print "lp213: skipping due to isinstance"
                    continue
                if verb: print "lP215:", repr(ops),repr(dir(ops))
                if hasattr(ops, prim):
                    if verb : print "lP216: hide happens"
                    if before:
                        rets = "${RED}(hidden by " + ops.__class__.__name__ +\
                            ")${NORMAL}"
                    else:
                        rets = '${GREEN}(hides "%s" from %s)${NORMAL}' %\
                            (prim, ops.__class__.__name__)
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
    
    def primitive_set_infostr(self, primsetname, cl, primlist):        
        sfull = getsourcefile(cl)
        sdir = os.path.dirname(sfull)
        sfil = os.path.basename(sfull)
        retstr = ""
        inherit = "None"
        inherit_list=[]
        for base in cl.__bases__:
            if base.__name__ in self.primsdict_kbn:
                inherit_list.append(base.__name__)
                inherit = "Yes"
        retstr += "\n  Inheritance      : ${BOLD}"+inherit+"${NORMAL}"
        if inherit is "Yes":
            #right now only works with one level of inheritence
            overrides_count = 0
            for prim in primlist:
                over = self.overrides(primsetname, prim)
                if over:
                    overrides_count += 1
            itot = 0
            for inherited in inherit_list:
                retstr + " (from ${BOLD}" + inherited + "${NORMAL}"
                if len(inherit_list) < 2:
                    iprimlist = self.primsdict_kbn[inherited]
                    itot = itot + (len(iprimlist) - overrides_count)
                    len_iprimlist = str(len(iprimlist) - overrides_count)
                    retstr += " (inherited ${BOLD}" + \
                        len_iprimlist + "${NORMAL} primitives"
                    retstr += " with ${BOLD}" + \
                        str(overrides_count)+" overridden)${NORMAL} "
        len_primlist = str(len(primlist))
        retstr += "\n  Local primitives :${BOLD} " + len_primlist + "${NORMAL}"
        if len(inherit_list) is 1:
            retstr += "\n  Total primitives : ${BOLD}"+ str(itot+len(primlist))\
                + "${NORMAL}"
        retstr += "\n  Source File      : ${BOLD}" + sfil + "${NORMAL}"
        retstr += "\n  Path             : ${BOLD}" + sdir + "${NORMAL}"
        return retstr

    def show_primitive_sets(self, return_string=False, prims=False):
        retstr = ""
        if not prims:
            retstr =  "\n" + "="*SW
            retstr += "\n${BOLD}PRIMITIVES BY SET${NORMAL}\n" + "="*SW 
            count = 1
        names = []
        for primset in self.primsets:
            nam = primset.__class__.__name__
            if nam in names:
                continue
            else:
                names.append(nam)
            if not prims:
                cl = self.name2class[nam]
                if len(self.primsets) == 1:
                    retstr += "\n\n  ${BOLD}%s${NORMAL}\n" % cl.astrotype
                else:
                    retstr += "\n\n%2d. ${BOLD}%s${NORMAL}\n" % (count,cl.astrotype)
                primlist = self.primsdict_kbn[nam]
                retstr += self.primitive_set_infostr(nam, cl, primlist)
                count += 1
            else:
                retstr += self.show_primitives(nam)
        retstr += "\n" + "${BOLD}=${NORMAL}"*SW
        if return_string:
            return retstr
        else:
            self.show(retstr)

    
    def show_primitives(self, primsetname, primset=None, i=0, indent=0, 
                        pdat=None, instance=None):
        INDENT = " "
        retstr = ""
        indentstr = INDENT*indent
        if primset == None:
            firstset = True
        else:
            firstset = False
        if firstset == True:
            primlist = self.primsdict_kbn[primsetname]
            primset = copy(primlist)
        else:
            myprimset = Set(self.primsdict_kbn[primsetname])
            givenprimset = Set(primset)
            prims = myprimset - givenprimset
            primlist = list(prims)
            primlist.sort()
            primset.extend(primlist)
        cl = self.name2class[primsetname]
        if firstset:           
            retstr = "\n\n${BOLD}" + "="*SW + "${NORMAL}"
            if self.show_info:
                retstr += "\n${BOLD}%s${NORMAL}" % (cl.astrotype)
                retstr += self.primitive_set_infostr(primsetname, cl, primlist) 
                retstr += "\n"
            else:
                retstr += "\n${BOLD}%s ${NORMAL}(%s)\n" % \
                    (cl.astrotype, primsetname)
            retstr += "${BOLD}=${NORMAL}"*SW 
            astrotype = cl.astrotype
            instance = self.class2instance[primsetname]
        else:
            if len(primlist) > 0:
                retstr += "\n"
                short = "(Following are inherited from "
                retstr += "${BLUE}%s%s%s)${NORMAL}"\
                    % (INDENT*indent, short, primsetname)
        if len(primlist) == 0:
            maxlenprim = 0
        else:
            maxlenprim = min(16, len(max(primlist, key=len)))
        for prim in primlist:
            i += 1
            hide = self.hides(primsetname, prim, instance = instance)
            over = self.overrides(primsetname, prim)
            primline = "\n%s%2d. %s" % (" "*indent, i, prim)
            pl = len(prim)
            if pl < maxlenprim:
                primline += " "*(maxlenprim-pl)
            if over:
                primline += "  ${BLUE}(overrides %s)${NORMAL}" % over
            if hide:
                primline += "  %s" % hide
            retstr += primline
            if self.show_usage:
                func = eval("instance." + prim)
                if hasattr(func, "pt_usage"):
                    retstr += " "*indent + "    ${YELLOW}DOC:" + \
                        eval("func.pt_usage") + "${NORMAL}"
                if hasattr(instance, "ptusage_" + prim):
                    retstr += " "*indent + "    ${YELLOW}DOC: " + \
                        eval("instance.ptusage_"+prim)+"${NORMAL}"
            if self.show_param:
                indent0 = indentstr+INDENT*5
                indent1 = indentstr+INDENT*6
                indentp = indent1+"parameter: "
                indentm = indent1+INDENT*2
                if pdat == None:
                    primsetinst = self.class2instance[primsetname]
                    paramdicttype = primsetinst.astrotype
                    paramdict = primsetinst.param_dict
                    pdat = (paramdicttype,paramdict)
                else:
                    paramdicttype = pdat[0]
                    paramdict = pdat[1]
                for primname in paramdict.keys():
                    if primname == prim:
                        if not firstset:
                            retstr += "\n" + term.GREEN + term.BOLD + indent0 \
                                + "(these parameter settings for an inherited" \
                                "primitive still originate in \n" + indent0 \
                                + " the ${BLUE}%s${GREEN} Primitive Set " \
                                "Parameters)${NORMAL}" % paramdicttype \
                                + term.NORMAL
                        paramnames = paramdict[primname].keys()
                        paramnames.sort()
                        for paramname in paramnames:
                            retstr += "\n" + term.GREEN + indentp + term.NORMAL + \
                                paramname + term.NORMAL
                            metadata = paramdict[primname][paramname].keys()
                            maxlen = len(max(metadata, key=len)) + 3
                            metadata.sort()
                            if "default" in metadata:
                                metadata.remove("default")
                                metadata.insert(0,"default")
                            for metadatum in metadata:
                                padding = " "*(maxlen - len(metadatum))
                                val = paramdict[primname][paramname][metadatum]
                                retstr += "\n" + term.GREEN + indentm + metadatum + \
                                padding + "= "+repr(val) + term.NORMAL

        for base in cl.__bases__:
            if base.__name__ in self.primsdict_kbn:
                retstr += self.show_primitives(base.__name__, primset=primset, i=i, 
                    indent=indent+2, pdat=pdat, instance=instance)
        return retstr
