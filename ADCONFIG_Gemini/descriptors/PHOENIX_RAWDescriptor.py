from astrodata import Lookups
from astrodata import Descriptors
import math

import astrodata
from astrodata.Calculator import Calculator

import GemCalcUtil 
from StandardPHOENIXKeyDict import stdkeyDictPHOENIX

class PHOENIX_RAWDescriptorCalc(Calculator):

    def __init__(self):
        pass
    
    def cwave(self, dataset, **args):
        """
        Return the cwave value for PHOENIX
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the central wavelength (nanometers)
        """
        try:
            hdu = dataset.hdulist
            retcwavefloat = hdu[0].header[stdkeyDictPHOENIX["key_phoenix_cwave"]]
        
        except KeyError:
            return None
        
        return float(retcwavefloat)
    
    def disperser(self, dataset, **args):
        """
        Return the disperser value for PHOENIX
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the disperser / grating used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retdisperserstring = hdu[0].header[stdkeyDictPHOENIX["key_phoenix_disperser"]]
        
        except KeyError:
            return None
        
        return str(retdisperserstring)
    
    def exptime(self, dataset, **args):
        """
        Return the exptime value for PHOENIX
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the total exposure time of the observation (seconds)
        """
        try:
            hdu = dataset.hdulist
            exptime = hdu[0].header[stdkeyDictPHOENIX["key_phoenix_exptime"]]
            coadds = hdu[0].header[stdkeyDictPHOENIX["key_phoenix_coadds"]]
            
            exptime = float(exptime) * float(coadds)
            return exptime
        
        except KeyError:
            return None
        
    def filtername(self, dataset, **args):
        """
        Return the filtername value for PHOENIX
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the unique filter identifier string
        """
        try:
            hdu = dataset.hdulist
            filter = hdu[0].header[stdkeyDictPHOENIX["key_phoenix_filter"]]

        except KeyError:
            return None
        
        return str(filter)
    
    def fpmask(self, dataset, **args):
        """
        Return the fpmask value for PHOENIX
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the focal plane mask used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retfpmaskstring = hdu[0].header[stdkeyDictPHOENIX["key_phoenix_fpmask"]]
        
        except KeyError:
            return None
                        
        return str(retfpmaskstring)

    def exptime(self, dataset, **args):
        """
        Return the exptime value for PHOENIX
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the total exposure time of the observation (seconds)
        """
        try:
            hdu = dataset.hdulist
            exptime = hdu[0].header[stdkeyDictPHOENIX["key_phoenix_exptime"]]
            coadds = hdu[0].header[stdkeyDictPHOENIX["key_phoenix_coadds"]]

            ret = float(exptime) * float(coadds)

            return ret

        except KeyError:
            return None

    def ra(self, dataset, **args):
        """
        Return the ra value for PHOENIX
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the ra in decimal degrees
        """
        try:
            hdu = dataset.hdulist
            ra = hdu[0].header[stdkeyDictPHOENIX["key_phoenix_ra"]]
            ra = GemCalcUtil.rasextodec(ra)
            return ra

        except KeyError:
            return None

    def dec(self, dataset, **args):
        """
        Return the dec value for PHOENIX
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the dec in decimal degrees
        """
        try:
            hdu = dataset.hdulist
            dec = hdu[0].header[stdkeyDictPHOENIX["key_phoenix_dec"]]
            dec = GemCalcUtil.degsextodec(dec)
            return dec

        except KeyError:
            return None

    def uttime(self, dataset, **args):
        try:
           hdu = dataset.hdulist
           uttimestring = hdu[0].header[stdkeyDictPHOENIX["key_phoenix_uttime"]]
           return uttimestring
        except KeyError:
            return None


    def disperser(self, dataset, **args):
        """
        Return the disperser value for PHOENIX
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the total exposure time of the observation (seconds)
        """

        return "Phoenix"
        # ThereIfixedIt

    def cwave(self, dataset, **args):
        """
        Return the cwave value for PHOENIX
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: cwave value

        This attempts to calculate the cwave from the grating position
        """
        try:
            hdu = dataset.hdulist
            cwavestring = hdu[0].header[stdkeyDictPHOENIX["key_phoenix_cwave"]]
            #m = re.match("encoder (-*\d*)", string)
            #if(m):
              #enc = m.group(1)
            # The phoenix grating equation, drop the cubic term for now
            # http://www.noao.edu/ngsc/phoenix/instruman.html#grating
            #c = 8573129-enc
            #b = -402584.5
            #a = 6355.8478
            #thing = sqrt((b*b)-(4*a*c))
            #theta = (-1.0*b + thing)/(2*a)
          
            try:
              cwave = float(cwavestring)
            except ValueError:
              cwave = None

            return cwave

        except KeyError:
            return None

        return cwavestring

