from astrodata import Lookups
from astrodata import Descriptors

from astrodata.Calculator import Calculator
from datetime import datetime
from time import strptime


import GemCalcUtil

stdkeyDictGMOS = {
    "key_gmos_ampname":"AMPNAME",
    "key_gmos_filter1":"FILTER1",
    "key_gmos_filter2":"FILTER2",
    "key_gmos_gain":"GAIN",
    }

class GMOS_RAWDescriptorCalc(Calculator):
    def __init__(self):
        # self.gmosamps = Lookups.getLookupTable("Gemini/GMOS/GMOSAmpTables", "gmosamps")
        # self.gmosampsBefore20060831 = Lookups.getLookupTable("Gemini/GMOS/GMOSAmpTables", "gmosampsBefore20060831")
        
        # slightly more efficiently, we can get both at once since they are in
        #  the same lookup space
        self.gmosamps,self.gmosampsBefore20060831 = Lookups.getLookupTable("Gemini/GMOS/GMOSAmpTables", "gmosamps", "gmosampsBefore20060831")
        
    def filtername(self, dataset):
        """Return the Filtername for GMOS_RAW data.
        @param dataset: the data set
        @type dataset: AstroData
        """
        try:
            hdu = dataset.hdulist
            #note: these calls could be to phuHeader now... None means not present
            filt1 = hdu[0].header[stdkeyDictGMOS["key_gmos_filter1"]]
            filt2 = hdu[0].header[stdkeyDictGMOS["key_gmos_filter2"]]
            filt1 = GemCalcUtil.removeComponentID(filt1)
            filt2 = GemCalcUtil.removeComponentID(filt2)
                        
            filts=[]
            if not "open" in filt1:
                filts.append(filt1)
            if not "open" in filt2:
                filts.append(filt2)
           
            if len(filts) == 0:
                return "open"
            filts.sort()                
            retfilt = "&".join(filts)
            
        except KeyError:
            return None
        
        return retfilt
     
    # used to get gain
    gmosamps = None

    def gain(self, dataset):
        """ 
        Return the gain for GMOS
        @rtype: list
        @returns: array of gain values, index 0 will be the gain of extension #1
                 (EXTNAME="SCI"and so on.
        """
        hdulist = dataset.hdulist
        # data is raw, not yet named:::: numsci = dataset.countExts("SCI")

        # initializations that should happen outside the loop
        ampinteg = dataset.phuHeader("AMPINTEG")
        datestr = dataset.phuHeader("DATE-OBS")
        obsdate = datetime(*strptime(datestr, "%Y-%m-%d")[0:6])
        oldampdate = datetime(2006,8,31,0,0)

        retary = []  
        for ext in dataset:
            # get the values
            gain = ext.header[stdkeyDictGMOS["key_gmos_gain"]]
            ampname = ext.header[stdkeyDictGMOS["key_gmos_ampname"]]
            # gmode
            if (gain > 3.0):
                gmode = "high"
            else:
                gmode = "low"

            # rmode
            if (ampinteg == None):
                rmode = "slow"
            else:
                if (ampinteg == 1000):
                    rmode = "fast"
                else:
                    rmode = "slow"

            gainkey = (rmode, gmode, ampname)
            
            try:
                if (obsdate > oldampdate):
                    gain = self.gmosamps[gainkey]
                else:
                    gain = self.gmosampsBefore20060831[gainkey]
            except KeyError:
                gain = None   
            retary.append(gain)       
 
        dataset.relhdul()

        return retary
        
    
    def gainorig( self, dataset ):
        '''
        
        
        '''
        hdulist = dataset.hdulist
        # data is raw, not yet named:::: numsci = dataset.countExts("SCI")

        # initializations that should happen outside the loop
        ampinteg = dataset.phuHeader("AMPINTEG")
        datestr = dataset.phuHeader("DATE-OBS")
        obsdate = datetime(*strptime(datestr, "%Y-%m-%d")[0:6])
        oldampdate = datetime(2006,8,31,0,0)

        retary = []  
        for ext in dataset:
            # get the values
            #gain = ext.header["GAINORIG"]
            if ext.header.has_key( 'GAINORIG' ):
                gain = ext.header["GAINORIG"]
            else:
                gain = ext.header["GAIN"]
            ampname = ext.header[stdkeyDictGMOS["key_gmos_ampname"]]
            # gmode
            if (gain > 3.0):
                gmode = "high"
            else:
                gmode = "low"

            # rmode
            if (ampinteg == None):
                rmode = "slow"
            else:
                if (ampinteg == 1000):
                    rmode = "fast"
                else:
                    rmode = "slow"

            gainkey = (rmode, gmode, ampname)
            
            try:
                if (obsdate > oldampdate):
                    gain = self.gmosamps[gainkey]
                else:
                    gain = self.gmosampsBefore20060831[gainkey]
            except KeyError:
                gain = None   
            retary.append(gain)       
 
        dataset.relhdul()

        return retary
        
    def ronorig( self, dataset ):
        '''
        
        '''
        # Epic klugin' right here.
        try:
            temp = dataset[1].header["RONORIG"]
        except:
            return self.fetchValue( "RDNOISE", dataset )
        
        return temp
            
    
