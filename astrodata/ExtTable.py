import os

import pyfits
import astrodata
from astrodata import Errors

class ExtTable(object):
    """
    ExtTable will create a dictionary structure keyed on 'EXTNAME' with an
    internal dictionary keyed on 'EXTVER' with ad as values.
    """
    def __init__(self, ad=None):
        self.xdict = {}
        if ad is None:
            print "WARNING: cannot create table without AstroData instance"
            self.ad = None
        if isinstance(ad, pyfits.core.HDUList):
            ad = astrodata.AstroData(ad)
        if not isinstance(ad, astrodata.AstroData):
            raise Errors.ExtTableError(\
                "Accepts only pyfits hdulist or AstroData instance")
        self.create_xdict(ad)

    def create_xdict(self, ad=None):
        if ad is None:
            raise Errors.ExtTableError("Accepts only one AstroData instance")
        extnames = []
        for hdu in ad.hdulist[1:]:
            xname = None
            xver = None
            if hdu.header.has_key('EXTNAME'):
                xname = hdu.header['EXTNAME']
                newname = True
                if xname in extnames:
                    newname=False
                extnames.append(xname)
            if hdu.header.has_key('EXTVER'):
                xver = hdu.header['EXTVER']
            if newname:
                self.xdict.update({xname:{xver:ad}})
            else:
                self.xdict[xname].update({xver:ad}) 

    def putAD(self, extname=None, extver=None, ad=None, auto_inc=False):
        if extname is None or ad is None:
            raise Errors.ExtTableError("At least extname and ad required")
        if extname in self.xdict.keys():
            if extver in self.xdict[extname].keys():
                # deal with collision
                if auto_inc is True:
                    extver_list = self.xdict[extname].keys()
                    extver_list.sort()
                    extver = extver_list.pop() + 1
                    self.xdict[extname].update({extver:ad})
                else:
                    raise Errors.ExtTableError(\
                        "Table already has %s, %s" % (extname, extver))
            else:
                # the extver is open, put in the AD!
                self.xdict[extname].update({extver:ad})
        else:
            #extname not in table, going to add it, then put in the AD!
            if extver is None and auto_inc:
                extver = 1
            self.xdict.update({extname:{extver:ad}})

    def getAD(self, extname=None, extver=None, asext=False):
        if extname is None or extver is None:
            raise Errors.ExtTableError("extname and extver are required")
        if extname in self.xdict.keys():
            if extver in self.xdict[extname].keys():
                if asext:
                    rad = self.xdict[extname][extver]
                    return rad[extname,extver]
                return self.xdict[extname][extver]
        print "Warning: Cannot find ad in %s, %s" % (extname,extver)
        return None
        
            
    def rows(self, asext=False):
        # find the largest extver out of all extnames
        bigver = 0
        for xnam in self.xdict.keys(): 
            for ver in self.xdict[xnam].keys():
                if ver > bigver:
                    bigver = ver
        index = 1
        # generator will keep yielding a table row until bigver hit
        while(index <= bigver):
            namlis = self.xdict.keys()
            rlist = []
            for xnam in self.xdict.keys():
                if index in self.xdict[xnam].keys():
                    if asext:
                        ad = self.xdict[xnam][index]
                        rlist.append((xnam,ad[xnam,index]))
                    else:
                        rlist.append((xnam,self.xdict[xnam][index]))
                else:
                    rlist.append((xnam, self.xdict[xnam][None]))
            yield rlist
            index += 1
                
    def largest_extver(self):
        bigver = 0
        for xnam in self.xdict.keys(): 
            for ver in self.xdict[xnam].keys():
                if ver > bigver:
                    bigver = ver
        return bigver




