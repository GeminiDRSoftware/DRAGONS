import pyfits
import os
import re
from AstroData import *
ldebug = False
verbose = False

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
                 showDescriptors = None, # string of comma separated descriptor names (function names!) 
                 # Now the descriptors, in alphabetical order
                 showairmass = False,
                 showcamera = False,
                 showcwave = False,
                 showdatasec = False,
                 showdetsec = False,
                 showdisperser = False,
                 showexptime = False,
                 showfiltername = False,
                 showfilterid = False,
                 showfpmask = False,
                 showgain = False,
                 showinstrument = False,
                 showmdfrow = False,
                 shownonlinear = False,
                 shownsciext = False,
                 showobject = False,
                 showobsmode = False,
                 showpixscale = False,
                 showrdnoise = False,
                 showsatlevel = False,
                 showutdate = False,
                 showuttime = False,
                 showwdelta = False,
                 showwrefpix = False,
                 showxbin = False,
                 showybin = False):
        """
        Recursively walk a given directory and put type information to stdout
        """
        global verbose
        global debug
        onlylist = only.split(",")
        if (verbose):
            print "onlylist:",repr(onlylist)
        
        verbose = False
        
        for root,dirn,files in os.walk(directory):
            if (verbose) :
                print "root:", root 
                print "dirn:", dirn
                
            if verbose:
                print "DS92:",root, repr(dirn), repr(file)
            if (".svn" not in root):
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
                            if verbose:
                                print "DS130:", repr(dtypes)
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
                                    if (verbose):
                                        print "DS148", tpname,repr(dtypes), "||", repr(ol)
                                    if (tpname in ol):
                                        found = True
                                    else:
                                        found = False

                            if (found != True):
                                continue

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
                            try:
                                if (showairmass == True):
                                    print "          airmass = %s" % str(fl.airmass())  
                                if (showcamera == True):
                                    print "          camera = %s" % str(fl.camera())  
                                if (showcwave == True):
                                    print "          cwave = %s" % str(fl.cwave())  
                                if (showdatasec == True):
                                    print "          datasec = %s" % str(fl.datasec())  
                                if (showdetsec == True):
                                    print "          detsec = %s" % str(fl.detsec())  
                                if (showdisperser == True):
                                    print "          disperser = %s" % str(fl.disperser())  
                                if (showexptime == True):
                                    print "          exptime = %s" % str(fl.exptime())  
                                if (showfiltername == True):
                                    print "          filtername = %s" % str(fl.filtername())  
                                if (showfilterid == True):
                                    print "          filterid = %s" % str(fl.filterid())  
                                if (showfpmask == True):
                                    print "          fpmask = %s" % str(fl.fpmask())  
                                if (showgain == True):
                                    print "          gain = %s" % str(fl.gain())  
                                if (showinstrument == True):
                                    print "          instrument = %s" % str(fl.instrument())  
                                if (showmdfrow == True):
                                    print "          mdfrow = %s" % str(fl.mdfrow())  
                                if (shownonlinear == True):
                                    print "          nonlinear = %s" % str(fl.nonlinear())  
                                if (shownsciext == True):
                                    print "          nsciext = %s" % str(fl.nsciext())
                                if (showobject == True):
                                    print "          object = %s" % str(fl.object())
                                if (showobsmode == True):
                                    print "          obsmode = %s" % str(fl.obsmode())
                                if (showpixscale == True):
                                    print "          pixscale = %s" % str(fl.pixscale())
                                if (showrdnoise == True):
                                    print "          rdnoise = %s" % str(fl.rdnoise())
                                if (showsatlevel == True):
                                    print "          satlevel = %s" % str(fl.satlevel())
                                if (showutdate == True):
                                    print "          utdate = %s" % str(fl.utdate())
                                if (showuttime == True):
                                    print "          uttime = %s" % str(fl.uttime())
                                if (showwdelta == True):
                                    print "          wdelta = %s" % str(fl.wdelta())
                                if (showwrefpix == True):
                                    print "          wrefpix = %s" % str(fl.wrefpix())
                                if (showxbin == True):
                                    print "          xbin = %s" % str(fl.xbin())
                                if (showybin == True):
                                    print "          ybin = %s" % str(fl.ybin())
                            except Descriptors.DescriptorExcept:
                                print "!!!! Descriptor Calculator Raised an Exception, possibly corrupt data"
                                raise

                            if (showDescriptors != None):
                                sdl = showDescriptors.split(",")
                                # print ol
                                for sd in sdl:
                                    # print "DS242:", sd
                                    try:
                                        dval = eval("fl."+sd+"()")
                                        print "          %s = %s" % (sd, str(dval))
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


        
   
