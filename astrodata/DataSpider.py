import pyfits
import os
import re
from AstroData import *
ldebug = False
verbose = False
from astrodata.adutils import terminal
from ReductionContextRecords import AstroDataRecord
import subprocess
uselocalcalserv = False
batchno = 100

if uselocalcalserv: # takes WAY TOO LONG~!!!!!!
    from astrodata.LocalCalibrationService import CalibrationService
    from CalibrationDefinitionLibrary import CalibrationDefinitionLibrary # For xml calibration requests

def shallowWalk(directory):
    global batchno
    opti = False
    if opti:
        print "sw: going to call os.listdir"
    ld = os.listdir(directory)
    if opti:
        print "sw: called os.listdir"
    root = directory
    dirn = []
    files = []
    if opti:
       print "sw: sorting directories from files in directory"

    if batchno != None:
        batchsize = batchno
    else:
        batchsize = 100

    for li in ld:
        if os.path.isdir(li):
            dirn.append(li)
        else:
            files.append(li)
        if len(files)> batchsize:
            if opti:
                print "yielding batch of " + str(batchsize)
                print repr(files)
            yield (root, [], files)
            files = []
    if opti:
        print "sw: yielding"
    yield (root, [], files)
    


class DataSpider(object):
    """
    DataSpider() is a work class to encapsulate
    reusable code to work the AstroData related classes.
    e.g. it will walk a directory using AstroData
    to check type sizes.   
    """
    hdulist = None
    contextType = None
    classificationLibrary = None
    calSearch = None
    def __init__(self, context = None):
        # ==== member vars ====
        self.contextType = context
        self.classificationLibrary = self.getClassificationLibrary()
        if uselocalcalserv:
            self.calService = CalibrationService()
            self.calDefLib = CalibrationDefinitionLibrary()
        
    def getClassificationLibrary(self):
        # @@todo: handle context here
        if (self.classificationLibrary == None):
            try:
                self.classificationLibrary = ClassificationLibrary()
            except CLAlreadyExists, s:
                self.classificationLibrary = s.clInstance
                
        return self.classificationLibrary
        
    def dumpinfo(self):
	    #print self.hdulist.info()
	    cards = self.hdulist[0].header.ascard

	    for hd in self.hdulist:
	        if (hd.data != None):
        	    try:
        	        print hd.data.type()
        	    except:
        	        print "Table"
                    
    def typewalk(self, directory = ".", only = "all", pheads = None,
                 showinfo = False,
                 onlyStatus = False,
                 onlyTypology = False,
                 # generic descriptors interface
                 showDescriptors = None, # string of comma separated descriptor names (function names!) 
                 filemask = None,
                 showCals = False,
                 incolog = True,
                 stayTop = False,
                 recipe = None,
                 raiseExcept = False,
                 where = None,
                 batchnum = None,
                 opti = None):
        """
        Recursively walk a given directory and put type information to stdout
        """
        global verbose
        global debug
        global batchno
        if batchnum != None:
            batchno = batchnum
            
        onlylist = only.split(",")
        if (verbose):
            print "onlylist:",repr(onlylist)
        
        verbose = False
        ldebug = False
        dirnum = 0
        if stayTop == True:
            walkfunc  = shallowWalk
            if opti:
                print "Doing a shallow walk"
        else:
            walkfunc = os.walk
            if opti:
                print "Doing an os.walk"
        for root,dirn,files in walkfunc(directory):
            if opti:
                print "Analyzing:", root
            dirnum += 1
            if (verbose) :
                print "root:", root 
                print "dirn:", dirn
                
            if verbose:
                print "DS92:",root, repr(dirn), repr(file)
            if (".svn" not in root):
                width = 10
                ## !!!!!
                ## !!!!! CREATE THE LINE WRITTEN FOR EACH DIRECTORY RECURSED !!!!!
                ## !!!!!
                fullroot = os.path.abspath(root)
                if root == ".":
                    rootln = "\n${NORMAL}${BOLD}directory: ${NORMAL}. ("+fullroot + ")${NORMAL}"
                else:
                    rootln = "\n${NORMAL}${BOLD}directory: ${NORMAL}"+root + "${NORMAL}"
                firstfile = True
                for tfile in files:
                    # we have considered removing this check in place of a
                    # pyfits open but that was not needed, the pyfits open
                    # is down lower, this is just to avoid checking files
                    # that are not named correctly to be FITS, so why check them?
                    # especially on a command recursing directories and potentially
                    # looking at a lot of files.
                    if filemask == None:
                        # @@NAMING: fits file mask for typewalk
                        mask = r".*?\.(fits|FITS)$"
                    else:
                        mask = filemask
                    try:
                        matched = re.match(mask, tfile)
                    except:
                        print "BAD FILEMASK (must be a valid regular expression):", mask
                        return str(sys.exc_info()[1])
                    if (re.match(mask, tfile)) :
                        if (ldebug) : print "FITS:", tfile

                        fname = os.path.join(root, tfile)
                       
                        try:
                            fl = AstroData(fname)
                        except ADExcept:
                            print "${RED}Could not open %s as AstroData${NORMAL}" %fname
                            continue

                        gain = 0
                        stringway = False
                        if (stringway):

                            if (onlyTypology == onlyStatus):
                                dtypes = self.classificationLibrary.discoverTypes(fname)
                            elif (onlyTypology):
                                dtypes = self.classificationLibrary.discoverTypology(fname)
                            elif (onlyStatus):
                                dtypes = self.classificationLibrary.discoverStatus(fname)

                        else:
                            # this is the AstroData Class way
                            # to ask the file itself

                            if (onlyTypology == onlyStatus):
                                dtypes = fl.discoverTypes()
                            elif (onlyTypology):
                                dtypes = fl.discoverTypology()
                            elif (onlyStatus):
                                dtypes = fl.discoverStatus()
                            if verbose:
                                print "DS130:", repr(dtypes)

                        # print "after classification"
                        if (dtypes != None) and (len(dtypes)>0):
                            #check to see if only is set
                            #only check for given type
                            found = False
                            if (only == "all"):
                                found=True
                            else:
                                # note: only can be split this way with no worry about
                                # whitespace because it's from the commandline, no whitespace
                                # allowed in that argument, just "," as a separator
                                ol = only.split(",")
                                # print ol
                                found = False
                                for tpname in dtypes:
                                    if (verbose):
                                        print "DS148", " in ", repr(ol),
                                    if (tpname in ol):
                                        found = True
                                        break
                                    if (verbose):
                                        print "yes, found = ", str(found)
                                        
                            if (found == True):
                                if where != None:
                                    # let them use underscore as spaces, bash + getopts doesn't like space in params even in quotes
                                    cleanwhere = re.sub("_"," ", where)
                                    ad = fl
                                    try:
                                        found = eval(cleanwhere)
                                    except:
                                        print "can't execute where:\n\t" + where + "\n\t" +cleanwhere
                                        
                                        print "reason:\n\t"+str(sys.exc_info()[1])+"\n"+repr(sys.exc_info())
                                        sys.exit(1)
                                        

                            if (found != True):
                                continue

                            if (firstfile == True):
                                print rootln
                            firstfile = False
                            
                            #  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            # !!!!PRINTING OUT THE FILE AND TYPE INFO!!!!
                            #  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            indent = 5
                            pwid = 40
                            fwid = pwid - indent 
                            # print start of string
                            
                            while len(tfile)>= fwid:
                                if False:
                                    part = tfile[:fwid]
                                    print "     ${BG_WHITE}%s${NORMAL}" % part
                                    tfile = tfile[fwid-1:]
                                else:
                                    print "     ${BG_WHITE}%s${NORMAL}" % tfile
                                    tfile = ""
                            
                            prlin = "     %s " % tfile
                            prlincolor = "     ${BG_WHITE}%s " % tfile
                            empty = " "*indent + "."*fwid
                            fwid = pwid+indent
                            lp = len(prlin)
                            nsp = pwid - ( lp % pwid )

                            # print out indent, filename, and "..." to justify types area"
                            # there is a way to do with with a comprehension?   

                            print prlincolor+("."*nsp)+"${NORMAL}",

                            # print dtypes
                            tstr = ""
                            termsize = terminal.getTerminalSize()
                            maxlen = termsize[0] - pwid -1
                            printed = False
                            dtypes.sort()
                            for dtype in dtypes:
                                if (dtype != None):
                                    newtype = "(%s) " % dtype
                                else:
                                    newtype = "(Unknown) "

                                # print "(%s)N20091027S0133.fits" % dtype ,
                                astr = tstr + newtype
                                if len(astr) >= maxlen:
                                    print "${BLUE}"+ tstr + "${NORMAL}"
                                    tstr = newtype
                                    print empty,
                                else:
                                    tstr = astr
                            if tstr != "":
                                print "${BLUE}"+ tstr + "${NORMAL}"
                                tstr = ""
                                astr = ""
                                printed = True

                            # new line at the end of the output
                            # print ""

                            if (showinfo == True):
                                hlist = pyfits.open(fname)
                                hlist.info()
                                hlist.close()

                            # print descriptors
                            # show descriptors                            
                            if (showDescriptors != None):
                                sdl = showDescriptors.split(",")
                                # print ol
                                # get maxlen
                                maxlen = 0
                                for sd in sdl:
                                    maxlen = max(len(sd),maxlen)
                                    
                                for sd in sdl:
                                    #print "DS242:", sd
                                    try:
                                        if "(" not in sd:
                                            dval = eval("fl."+sd+"()")
                                        else:
                                            dval = eval("fl."+sd)
                                        pad = " " * (maxlen - len(sd))
                                        sd = str(sd) + pad
                                        print ("          ${BOLD}%s${NORMAL} = %s") % (sd, str(dval))
                                        
                                    except:
                                        pad = " " * (maxlen - len(sd))
                                        sd = str(sd) + pad
                                        print ("          ${BOLD}%s${NORMAL} = ${RED}FAILED${NORMAL}: %s") % (sd, str(sys.exc_info()[1])) 
                                        if raiseExcept:
                                            raise
                                        
                                        

                            # if phead then there are headers to print per file
                            if (pheads != None):
                                #print "          -----------"sys.exec
                                print "          ${UNDERLINE}PHU Headers${NORMAL}"
                                #print "          -----------"
                                #print "pheads", pheads  
                                hlist = pyfits.open(fname)
                                pheaders = pheads.split(",")
                                for headkey in pheaders:
                                    #if in phu, this is the code

                                    try:
                                        print "            %s = (%s)" % (headkey, hlist[0].header[headkey])
                                    except KeyError:
                                        print "            %s not present in PHU of %s" % (headkey, tfile) 

                                hlist.close()
                            if (showCals == True):
                                adr = AstroDataRecord(fl)
                                for caltyp in ["bias", "twilight"]:
                                    rq = self.calDefLib.getCalReq([adr],caltyp)[0]
                                    try:
                                        cs = "%s" % (str(self.calService.search(rq)[0]))
                                    except:

                                        cs = "No %s found, %s " % ( caltyp, str(sys.exc_info()[1]))
                                        raise
                                    print "          %10s: %s" % (caltyp, cs)
                            if (recipe):
                                banner = ' Running Recipe "%s" on %s ' % (recipe, fname)
                                print "${REVERSE}${RED}" + " "*len(banner)
                                print banner
                                print " "*len(banner)+"${NORMAL}"
                                
                                if recipe == "default":
                                    rs = ""
                                else:
                                    rs = "-r %s" % recipe
                                subprocess.call("reduce %s %s" % (rs, fname), shell=True)
                    else:
                        if (verbose) : print "%s is not a FITS file" % tfile
                    
            if False: # done with walk function switching if stayTop == True:
                # cheap way to not recurse.
                break;

        
   
