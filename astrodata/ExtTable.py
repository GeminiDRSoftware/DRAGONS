import os

import pyfits
import astrodata
from astrodata import Errors

class ExtTable(object):
    """
    ExtTable will create a dictionary structure keyed on 'EXTNAME' with an
    internal dictionary keyed on 'EXTVER' with ad as values.
    """
    def __init__(self, ad=None, hdul=None):
        self.xdict = {}
        if ad and hdul:
            raise Errors.ExtTableError("Can only take ad OR hdul not both")
        if ad is None and hdul is None:
            raise Errors.ExtTableError("Object requires AstroData OR hdulist")
        self.ad = ad
        self.hdul = hdul
        self.create_xdict()

    def create_xdict(self):
        hdulist = None
        if isinstance(self.hdul, pyfits.core.HDUList):
            hdulist = self.hdul
        elif isinstance(self.ad, astrodata.AstroData):
            hdulist = self.ad.hdulist
        extnames = []
        for i in range(1,len(hdulist)):
            xname = None
            xver = None
            hdu = hdulist[i]
            if 'EXTNAME' in hdu.header:
                xname = hdu.header['EXTNAME']
                newname = True
                if xname in extnames:
                    newname=False
                else:
                    extnames.append(xname)
            if 'EXTVER' in hdu.header:
                xver = hdu.header['EXTVER']
            if newname:
                if self.ad is None:
                    self.xdict.update({xname:{xver:(True,i)}})
                else:
                    self.xdict.update({xname:{xver:(self.ad,i)}})
            else:
                if self.ad is None:
                    self.xdict[xname].update({xver:(True,i)})
                else:
                    self.xdict[xname].update({xver:(self.ad,i)}) 

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




