from astrodata import Lookups
from astrodata import Descriptors

from astrodata.Calculator import Calculator

import GemCalcUtil

from StandardGEMINIKeyDict import stdkeyDictGEMINI
from Generic_Descriptor import Generic_DescriptorCalc

class GEMINI_DescriptorCalc(Generic_DescriptorCalc):
    
    def airmass(self, dataset, **args):
        """
        Return the airmass value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the mean airmass for the observation
        """
        try:
            hdu = dataset.hdulist
            retairmassfloat = hdu[0].header[stdkeyDictGEMINI["key_gemini_airmass"]]

        except KeyError:
            return None
        
        return float(retairmassfloat)
    
    def azimuth(self, dataset, **args):
        """
        Return the azimuth value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the azimuth of the observation
        """
        try:
            hdu = dataset.hdulist
            retazimuthfloat = hdu[0].header[stdkeyDictGEMINI["key_gemini_azimuth"]]

        except KeyError:
            return None
        
        return float(retazimuthfloat)
    
    def crpa(self, dataset, **args):
        """
        Return the current cass rotator angle value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the current cass rotator angle of the observation
        """
        try:
            hdu = dataset.hdulist
            retcrpafloat = hdu[0].header[stdkeyDictGEMINI["key_gemini_crpa"]]

        except KeyError:
            return None
        
        return float(retcrpafloat)
    
    def datalab(self, dataset, **args):
        """
        Return the DHS data label value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the DHS data label of the observation
        """
        try:
            hdu = dataset.hdulist
            retdatalabstring = hdu[0].header[stdkeyDictGEMINI["key_gemini_datalab"]]

        except KeyError:
            return None
        
        return str(retdatalabstring)
    
    def dec(self, dataset, **args):
        """
        Return the declination value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the declination of the observation
        """
        try:
            hdu = dataset.hdulist
            retdecfloat = hdu[0].header[stdkeyDictGEMINI["key_gemini_dec"]]

        except KeyError:
            return None
        
        return float(retdecfloat)
    
    def elevation(self, dataset, **args):
        """
        Return the elevation value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the elevation of the observation
        """
        try:
            hdu = dataset.hdulist
            retelevationfloat = hdu[0].header[stdkeyDictGEMINI["key_gemini_elevation"]]

        except KeyError:
            return None
        
        return float(retelevationfloat)
    
    def gemprgid(self, dataset, **args):
        """
        Return the Gemini science program ID value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the Gemini science program ID of the observation
        """
        try:
            hdu = dataset.hdulist
            retgemprgidstring = hdu[0].header[stdkeyDictGEMINI["key_gemini_gemprgid"]]

        except KeyError:
            return None
        
        return str(retgemprgidstring)
    
    def obsclass(self, dataset, **args):
        """
        Return the observation class value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the class of the observation
        """
        try:
            hdu = dataset.hdulist
            retobsclassstring = hdu[0].header[stdkeyDictGEMINI["key_gemini_obsclass"]]

        except KeyError:
            return None
        
        return str(retobsclassstring)
    
    def obsid(self, dataset, **args):
        """
        Return the observation ID / data label value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the observation ID / data label of the observation
        """
        try:
            hdu = dataset.hdulist
            retobsidstring = hdu[0].header[stdkeyDictGEMINI["key_gemini_obsid"]]

        except KeyError:
            return None
        
        return str(retobsidstring)
    
    def obstype(self, dataset, **args):
        """
        Return the observation type value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the type of the observation
        """
        try:
            hdu = dataset.hdulist
            retobstypestring = hdu[0].header[stdkeyDictGEMINI["key_gemini_obstype"]]

        except KeyError:
            return None
        
        return str(retobstypestring)
    
    def ra(self, dataset, **args):
        """
        Return the right ascension value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the right ascension of the observation
        """
        try:
            hdu = dataset.hdulist
            retrafloat = hdu[0].header[stdkeyDictGEMINI["key_gemini_ra"]]

        except KeyError:
            return None
        
        return float(retrafloat)
    
    def rawbg(self, dataset, **args):
        """
        Return the raw background value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the raw background of the observation
        """
        try:
            hdu = dataset.hdulist
            retrawbgstring = hdu[0].header[stdkeyDictGEMINI["key_gemini_rawbg"]]

        except KeyError:
            return None
        
        return str(retrawbgstring)
    
    def rawcc(self, dataset, **args):
        """
        Return the raw cloud cover value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the raw cloud cover of the observation
        """
        try:
            hdu = dataset.hdulist
            retrawccstring = hdu[0].header[stdkeyDictGEMINI["key_gemini_rawcc"]]

        except KeyError:
            return None
        
        return str(retrawccstring)
    
    def rawgemqa(self, dataset, **args):
        """
        Return the raw Gemini quality assesment value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the raw Gemini quality assesment of the observation
        """
        try:
            hdu = dataset.hdulist
            retrawgemqastring = hdu[0].header[stdkeyDictGEMINI["key_gemini_rawgemqa"]]

        except KeyError:
            return None
        
        return str(retrawgemqastring)
    
    def rawiq(self, dataset, **args):
        """
        Return the raw image quality value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string 
        @return: the raw image quality of the observation
        """
        try:
            hdu = dataset.hdulist
            retrawiqstring = hdu[0].header[stdkeyDictGEMINI["key_gemini_rawiq"]]

        except KeyError:
            return None
        
        return str(retrawiqstring)
    
    def rawpireq(self, dataset, **args):
        """
        Return the raw PI requirements value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string 
        @return: whether the PI requirement were met
        """
        try:
            hdu = dataset.hdulist
            retrawpireqstring = hdu[0].header[stdkeyDictGEMINI["key_gemini_rawpireq"]]

        except KeyError:
            return None
        
        return str(retrawpireqstring)
    
    def rawwv(self, dataset, **args):
        """
        Return the raw water vapour / transparency value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string 
        @return: the raw water vapour / transparency of the observation
        """
        try:
            hdu = dataset.hdulist
            retrawwvstring = hdu[0].header[stdkeyDictGEMINI["key_gemini_rawwv"]]

        except KeyError:
            return None
        
        return str(retrawwvstring)
    
    def ssa(self, dataset, **args):
        """
        Return the Gemini SSA value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the Gemini SSA for the observation
        """
        try:
            hdu = dataset.hdulist
            retssastring = hdu[0].header[stdkeyDictGEMINI["key_gemini_ssa"]]

        except KeyError:
            return None
        
        return str(retssastring)
    
    def uttime(self, dataset, **args):
        """
        Return the uttime value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the UT at the start of the observation (HH:MM:SS.S)
        """
        try:
            hdu = dataset.hdulist
            retuttimestring = hdu[0].header[stdkeyDictGEMINI["key_gemini_uttime"]]
        
        except KeyError:
            return None
        
        return str(retuttimestring)
    
    def xoffset(self, dataset, **args):
        """
        Return the telescope offset in x in arcsec value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the telescope offset in x in arcsec of the observation
        """
        try:
            hdu = dataset.hdulist
            retxoffsetfloat = hdu[0].header[stdkeyDictGEMINI["key_gemini_xoffset"]]

        except KeyError:
            return None
        
        return float(retxoffsetfloat)

    def yoffset(self, dataset, **args):
        """
        Return the telescope offset in y in arcsec value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the telescope offset in y in arcsec of the observation
        """
        try:
            hdu = dataset.hdulist
            retyoffsetfloat = hdu[0].header[stdkeyDictGEMINI["key_gemini_yoffset"]]

        except KeyError:
            return None
        
        return float(retyoffsetfloat)
