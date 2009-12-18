import pyfits
import os
import re
from AstroData import *
ldebug = False
verbose = False

from FunctDescDict import functionDict

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
       
    def __init__(self, context = None):
        # ==== member vars ====
        self.contextType = context
        self.classificationLibrary = self.getClassificationLibrary()
        
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
                 showDescriptors = None): # string of comma separated descriptor names (function names!) 
        """
        Recursively walk a given directory and put type information to stdout
        """

        onlylist = only.split(",")
        #print onlylist
        
        
        for root,dirn,files in os.walk(directory):
            if (verbose) :
                print "root:", root 
                print "dirn:", dirn
            width = 10
            rootln = "\ndirectory: "+root
            firstfile = True
            for tfile in files:
                # we have considered removing this check in place of a
                # pyfits open but that was not needed, the pyfits open
                # is down lower, this is just to avoid checking files
                # that are not named correctly to be FITS, so why check them?
                # especially on a command recursing directories and potentially
                # looking at a lot of files.
                if (re.match(r".*?\.(fits|FITS)", tfile)) :
                    if (ldebug) : print "FITS:", tfile
                  
                    fname = os.path.join(root, tfile)
                    
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

                        fl = AstroData(fname)
                        if (onlyTypology == onlyStatus):
                            dtypes = fl.discoverTypes()
                        elif (onlyTypology):
                            dtypes = fl.discoverTypology()
                        elif (onlyStatus):
                            dtypes = fl.discoverStatus()
                        
                        fl.close()
                    
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
                            for tpname in dtypes:
                                if (tpname in ol):
                                    found = True
                            
                        if (found != True):
                            break
                        
                        if (firstfile == True):
                            print rootln
                        firstfile = False
                        prlin = "     %s" % tfile
                        pwid = 40
                        lp = len(prlin)
                        nsp = pwid - ( lp % pwid )
                        
                        # there is a way to do with with a comprehension?   
                        sp ="........."
                        sp = sp + sp + sp
                        print prlin,sp[:nsp],
                            
                        # print dtypes
                        for dtype in dtypes:
                            if (dtype != None):
                                print "(%s)" % dtype ,
                            else:
                                print "Unknown",

                        # new line at the end of the output
                        print ""
                        
                        if (showinfo == True):
                            hlist = pyfits.open(fname)
                            hlist.info()
                            hlist.close()
                        
                        # print descriptors
                        fl = AstroData(fname)
                            
                        if (showDescriptors != None):
                            sdl = showDescriptors.split(",")
                            # print ol
                            for sd in sdl:
                                print "DS242:", sd # sd = function name
                                try:
                                    dval = eval("fl."+sd+"()")
                                    # print actual descriptor name
                                    print "          %s = %s" % (functionDict[sd], str(dval))
                                except:
                                    print "Failed Descriptor Calculation for %s" % sd
                                    raise
                        
                        # if phead then there are headers to print per file
                        if (pheads != None):
                            print "          -----------"
                            print "          PHU Headers"
                            print "          -----------"
                            #print "pheads", pheads  
                            hlist = pyfits.open(fname)
                            pheaders = pheads.split(",")
                            for headkey in pheaders:
                                #if in phu, this is the code
                                
                                try:
                                    print "          %s = (%s)" % (headkey, hlist[0].header[headkey])
                                except KeyError:
                                    print "          %s not present in PHU of %s" % (headkey, tfile) 
                            print "          -----------"

                            hlist.close()
                else:
                    if (verbose) : print "%s is not a FITS file" % tfile
                    
    
        
   
