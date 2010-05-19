from astrodata import Lookups
from astrodata import Descriptors
import re
import traceback

from astrodata.Calculator import Calculator

from datetime import datetime
from time import strptime

import GemCalcUtil

from StandardGMOSKeyDict import stdkeyDictGMOS
from StandardGenericKeyDict import stdkeyDictGeneric
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class GMOS_DescriptorCalc(GEMINI_DescriptorCalc):
    
    gmosampsGain = None
    gmosampsGainBefore20060831 = None
    gmosampsRdnoise = None
    gmosampsRdnoiseBefore20060831 = None
    
    def __init__(self):
        # self.gmosampsGain = \
        #     Lookups.getLookupTable('Gemini/GMOS/GMOSAmpTables',
        #                            'gmosampsGain')
        # self.gmosampsGainBefore20060831 = \
        #     Lookups.getLookupTable('Gemini/GMOS/GMOSAmpTables',
        #                            'gmosampsGainBefore20060831')
        
        # slightly more efficiently, we can get both at once since they are in
        # the same lookup space
        self.gmosampsGain, self.gmosampsGainBefore20060831 = \
            Lookups.getLookupTable('Gemini/GMOS/GMOSAmpTables',
                                   'gmosampsGain',
                                   'gmosampsGainBefore20060831')
        self.gmosampsRdnoise, self.gmosampsRdnoiseBefore20060831 = \
            Lookups.getLookupTable('Gemini/GMOS/GMOSAmpTables',
                                   'gmosampsRdnoise',
                                   'gmosampsRdnoiseBefore20060831')
    
    def amproa(self, dataset, asList=False, **args):
        """
        Return the amproa value for GMOS
        This is a composite string containing the name of the detector
        amplifier (ampname) and the readout area of that ccd (detsec). If
        asList = True, a list is returned, where the number of array elements
        equals the number of pixel data extensions in the image.
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string or list (if asList = True)
        @return: the combined detector amplifier name and readout area
        """
        try:
            if (asList):
                retamproa = []
                for ext in dataset['SCI']:
                    ampname = ext.header[stdkeyDictGMOS['key_gmos_ampname']]
                    detsec = ext.header[stdkeyDictGMOS['key_gmos_detsec']]
                    retamproa.append("'%s':%s" % (ampname, detsec))
            
            else:
                try:
                    if (dataset.nsciext() < 1):
                        hdu = dataset.hdulist
                        ampname = \
                            hdu[1].header[stdkeyDictGMOS['key_gmos_ampname']]
                        detsec = \
                            hdu[1].header[stdkeyDictGMOS['key_gmos_detsec']]
                        retamproa = ("'%s':%s" % (ampname, detsec))
                    else:
                        raise Exception('Please use asList=True to obtain a list')
                
                except Exception:
                    print traceback.format_exc()
        
        except KeyError:
            return None
        
        return retamproa
    
    def camera(self, dataset, **args):
        """
        Return the camera value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the camera used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retcamerastring = hdu[0].header[stdkeyDictGMOS['key_gmos_camera']]
        
        except KeyError:
            return None
        
        return str(retcamerastring)
    
    def cwave(self, dataset, **args):
        """
        Return the cwave value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the central wavelength (micrometers)
        """
        try:
            hdu = dataset.hdulist
            cwave = float(hdu[0].header[stdkeyDictGMOS['key_gmos_cwave']])
            retcwavefloat = cwave / 1000.
        
        except KeyError:
            return None
        
        return float(retcwavefloat)
    
    def datasec(self, dataset, asList=False, **args):
        """
        Return the datasec value for GMOS
        If asList = True, a list is returned, where the number of array
        elements equals the number of pixel data extensions in the image.
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string or list (if asList = True)
        @returns: the data section
        """
        try:
            if (asList):
                retdatasec = []
                for ext in dataset['SCI']:
                    datasec = ext.header[stdkeyDictGMOS['key_gmos_datasec']]
                    retdatasec.append(datasec)
            
            else:
                try:
                    if (dataset.nsciext() < 1):
                        hdu = dataset.hdulist
                        retdatasec = \
                            hdu[1].header[stdkeyDictGMOS['key_gmos_datasec']]
                    else:
                        raise Exception('Please use asList=True to obtain a list')
                
                except Exception:
                    print traceback.format_exc()
        
        except KeyError:
            return None
        
        return retdatasec
    
    def detroa(self, dataset, asList=False, **args):
        """
        Return the detroa value for GMOS
        This is a composite string containing the name of the detector
        (ccdname) and the readout area of that detector (detsec). If
        asList = True, a list is returned, where the number of array elements
        equals the number of pixel data extensions in the image.
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string of list (if asList = True)
        @return: the combined detector name and readout area
        """
        try:
            if (asList):
                retdetroa = []
                for ext in dataset['SCI']:
                    ccdname = ext.header[stdkeyDictGMOS['key_gmos_ccdname']]
                    detsec = ext.header[stdkeyDictGMOS['key_gmos_detsec']]
                    retdetroa.append("'%s':%s" % (ccdname, detsec))
            
            else:
                try:
                    if (dataset.nsciext() < 1):
                        hdu = dataset.hdulist
                        ccdname = \
                            hdu[1].header[stdkeyDictGMOS['key_gmos_ccdname']]
                        detsec = \
                            hdu[1].header[stdkeyDictGMOS['key_gmos_detsec']]
                        retdetroa = ("'%s':%s" % (ccdname, detsec))
                    else:
                        raise Exception('Please use asList=True to obtain a list')
                
                except Exception:
                    print traceback.format_exc()
        
        except KeyError:
            return None
        
        return retdetroa
    
    def detsec(self, dataset, asList=False, **args):
        """
        Return the detsec value for GMOS
        If asList = True, a list is returned, where the number of array
        elements equals the number of pixel data extensions in the image.
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string or list (if asList = True)
        @returns: the detector section
        """
        try:
            if (asList):
                retdetsec = []
                for ext in dataset['SCI']:
                    detsec = ext.header[stdkeyDictGMOS['key_gmos_detsec']]
                    retdetsec.append(detsec)
            
            else:
                try:
                    if (dataset.nsciext() < 1):
                        hdu = dataset.hdulist
                        retdetsec = \
                            hdu[1].header[stdkeyDictGMOS['key_gmos_detsec']]
                    else:
                        raise Exception('Please use asList=True to obtain a list')
                
                except Exception:
                    print traceback.format_exc()
        
        except KeyError:
            return None
        
        return retdetsec
    
    def disperser(self, dataset, stripID=False, pretty=False, **args):
        """
        Return the disperser value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the disperser / grating used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            disperser = hdu[0].header[stdkeyDictGMOS['key_gmos_disperser']]
            
            if (pretty):
                # In the case of GMOS, pretty is stripID with additionally the
                # '+' removed from the string
                stripID = True
            
            if (stripID):
                if (pretty):
                    retdisperserstring = \
                        GemCalcUtil.removeComponentID(disperser).strip('+')
                else:
                    retdisperserstring = \
                        GemCalcUtil.removeComponentID(disperser)
            else:
                retdisperserstring = disperser
        
        except KeyError:
            return None
        
        return str(retdisperserstring)
    
    def exptime(self, dataset, **args):
        """
        Return the exptime value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the total exposure time of the observation (seconds)
        """
        try:
            hdu = dataset.hdulist
            exptime = hdu[0].header[stdkeyDictGMOS['key_gmos_exptime']]
            
            # Sanity check for times when the GMOS DC is stoned
            if ((exptime > 10000.) or (exptime < 0.)):
                return None
            else:
                retexptimefloat = exptime
        
        except KeyError:
            return None
        
        return float(retexptimefloat)
    
    def filterid(self, dataset, **args):
        """
        Return the filterid value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the unique filter ID number string
        """
        try:
            hdu = dataset.hdulist
            filtid1 = str(hdu[0].header[stdkeyDictGMOS['key_gmos_filtid1']])
            filtid2 = str(hdu[0].header[stdkeyDictGMOS['key_gmos_filtid2']])
            
            filtsid = []
            filtsid.append(filtid1)
            filtsid.append(filtid2)
            filtsid.sort()
            retfilteridstring = '&'.join(filtsid)
        
        except KeyError:
            return None
        
        return str(retfilteridstring)
    
    def filtername(self, dataset, stripID=False, pretty=False, **args):
        """
        Return the filtername value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the unique filter identifier string
        """
        try:
            hdu = dataset.hdulist
            filter1 = hdu[0].header[stdkeyDictGMOS['key_gmos_filter1']]
            filter2 = hdu[0].header[stdkeyDictGMOS['key_gmos_filter2']]
            
            if (pretty):
                stripID = True
            
            if (stripID):
                filter1 = GemCalcUtil.removeComponentID(filter1)
                filter2 = GemCalcUtil.removeComponentID(filter2)
            
            filters = []
            if not 'open' in filter1:
                filters.append(filter1)
            if not 'open' in filter2:
                filters.append(filter2)
            
            if len(filters) == 0:
                retfilternamestring = 'open'
            else:
                retfilternamestring = '&'.join(filters)
        
        except KeyError:
            return None
        
        return str(retfilternamestring)
    
    def fpmask(self, dataset, **args):
        """
        Return the fpmask value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the focal plane mask used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            fpmask = hdu[0].header[stdkeyDictGMOS['key_gmos_fpmask']]
            
            if fpmask == 'None':
                retfpmaskstring = 'Imaging'
            else:
                retfpmaskstring = fpmask
        
        except KeyError:
            return None
        
        return str(retfpmaskstring)
    
    def gain(self, dataset, asList=False, **args):
        """
        Return the gain value for GMOS
        If asList = True, a list is returned, where the number of array
        elements equals the number of pixel data extensions in the image.
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float or list (if asList = True)
        @returns: the gain in electrons/ADU
        """
        try:
            hdu = dataset.hdulist
            ampinteg = hdu[0].header[stdkeyDictGMOS['key_gmos_ampinteg']]
            utdate = hdu[0].header[stdkeyDictGeneric['key_generic_utdate']]
            obsutdate = datetime(*strptime(utdate, '%Y-%m-%d')[0:6])
            oldutdate = datetime(2006, 8, 31, 0, 0)

            if (asList):
                retgain = []
                for ext in dataset['SCI']:
                    # Descriptors must work for all AstroData Types so check
                    # if the original gain keyword exists to use for the
                    # look-up table
                    if ext.header.has_key(stdkeyDictGMOS['key_gmos_gainorig']):
                        headergain = \
                            ext.header[stdkeyDictGMOS['key_gmos_gainorig']]
                    else:
                        headergain = \
                            ext.header[stdkeyDictGMOS['key_gmos_gain']]
                    
                    ampname = ext.header[stdkeyDictGMOS['key_gmos_ampname']]
                    gmode = dataset.gainmode()
                    rmode = dataset.readspeedmode()
                    
                    gainkey = (rmode, gmode, ampname)
                    
                    try:
                        if (obsutdate > oldutdate):
                            gain = self.gmosampsGain[gainkey]
                        else:
                            gain = self.gmosampsGainBefore20060831[gainkey]
                    
                    except KeyError:
                        gain = None
                    
                    retgain.append(gain)
            
            else:
                try:
                    if (dataset.nsciext() < 1):
                        # Descriptors must work for all AstroData Types so
                        # check if the original gain keyword exists to use for
                        # the look-up table
                        if hdu[1].header.has_key(stdkeyDictGMOS['key_gmos_gainorig']):
                            headergain = \
                                hdu[1].header[stdkeyDictGMOS['key_gmos_gainorig']]
                        else:
                            headergain = \
                                hdu[1].header[stdkeyDictGMOS['key_gmos_gain']]
                        
                        ampname = \
                            hdu[1].header[stdkeyDictGMOS['key_gmos_ampname']]
                        gmode = dataset.gainmode()
                        rmode = dataset.readspeedmode()
                        
                        gainkey = (rmode, gmode, ampname)
                        
                        try:
                            if (obsutdate > oldutdate):
                                retgain = self.gmosampsGain[gainkey]
                            else:
                                retgain = \
                                    self.gmosampsGainBefore20060831[gainkey]
                        
                        except KeyError:
                            return None
                    
                    else:
                        raise Exception('Please use asList=True to obtain a list')
                
                except Exception:
                    print traceback.format_exc()
        
        except KeyError:
            return None
        
        return retgain
    
    gmosampsGain = None
    gmosampsGainBefore20060831 = None
    
    def gainmode(self, dataset, **args):
        """
        Return the gain mode value for GMOS
        This is used in the gain descriptor for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the gain mode
        """
        try:
            hdu = dataset.hdulist
            # Descriptors must work for all AstroData Types so check
            # if the original gain keyword exists to use for the look-up table
            if hdu[1].header.has_key(stdkeyDictGMOS['key_gmos_gainorig']):
                headergain = \
                    hdu[1].header[stdkeyDictGMOS['key_gmos_gainorig']]
            else:
                headergain = \
                    hdu[1].header[stdkeyDictGMOS['key_gmos_gain']]
            
            if (headergain > 3.0):
                retgainmodestring = 'high'
            else:
                retgainmodestring = 'low'
        
        except KeyError:
            return None
        
        return retgainmodestring
    
    def mdfrow(self, dataset, asList=False, **args):
        """
        Return the mdfrow value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the corresponding reference row in the MDF
        """
        try:
            if (asList):
                retmdfrow = []
                for ext in dataset['SCI']:
                    mdfrow = ext.header[stdkeyDictGMOS['key_gmos_mdfrow']]
                    retmdfrow.append(mdfrow)
            
            else:
                try:
                    if (dataset.nsciext() < 1):
                        hdu = dataset.hdulist
                        retmdfrow = \
                            hdu[1].header[stdkeyDictGMOS['key_gmos_mdfrow']]
                    else:
                        raise Exception('Please use asList=True to obtain a list')
                
                except Exception:
                    print traceback.format_exc()
        
        except KeyError:
            return None
        
        return retmdfrow
    
    def nsciext(self, dataset, **args):
        """
        Return the nsciext value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the number of science extensions
        """
        retnsciextint = dataset.countExts('SCI')
        
        return int(retnsciextint)
    
    def obsmode(self, dataset, **args):
        """
        Return the obsmode value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the observing mode
        """
        try:
            hdu = dataset.hdulist
            masktype = hdu[0].header[stdkeyDictGMOS['key_gmos_masktype']]
            maskname = hdu[0].header[stdkeyDictGMOS['key_gmos_maskname']]
            grating = hdu[0].header[stdkeyDictGMOS['key_gmos_disperser']]
            
            if masktype == 0:
                retobsmodestring = 'IMAGE'
            
            elif masktype == -1:
                retobsmodestring = 'IFU'
            
            elif masktype == 1:
                
                if re.search('arcsec', maskname) != None and \
                    re.search('NS', maskname) == None:
                    retobsmodestring = 'LONGSLIT'
                else:
                    retobsmodestring = 'MOS'
            
            else:
                # if obsmode cannot be determined, set it equal to IMAGE
                # instead of crashing
                retobsmodestring = 'IMAGE'
            
            # mask or IFU cannot be used without grating
            if grating == 'MIRROR' and masktype != 0:
                retobsmodestring == 'IMAGE' 
        
        except KeyError:
            return None
        
        return str(retobsmodestring)
    
    def pixscale(self, dataset, **args):
        """
        Return the pixscale value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the pixel scale (arcsec/pixel)
        """
        try:
            hdu = dataset.hdulist
            instrument = \
                hdu[0].header[stdkeyDictGeneric['key_generic_instrument']]
            # Assume ccdsum is the same in all extensions
            ccdsum = hdu[1].header[stdkeyDictGMOS['key_gmos_ccdsum']]
            
            if instrument == 'GMOS-N':
                scale = 0.0727
            if instrument == 'GMOS-S':
                scale = 0.073
            
            if ccdsum != None:
                xccdbin, yccdbin = ccdsum.split()
                retpixscalefloat = float(yccdbin) * scale
            else:
                retpixscalefloat = scale
        
        except KeyError:
            return None
        
        return float(retpixscalefloat)
    
    def rdnoise(self, dataset, asList=False, **args):
        """
        Return the rdnoise value for GMOS
        If asList = True, a list is returned, where the number of array
        elements equals the number of pixel data extensions in the image.
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float or list (if asList = True)
        @returns: the estimated readout noise values (electrons)
        """
        try:
            hdu = dataset.hdulist
            ampinteg = hdu[0].header[stdkeyDictGMOS['key_gmos_ampinteg']]
            utdate = hdu[0].header[stdkeyDictGeneric['key_generic_utdate']]
            obsutdate = datetime(*strptime(utdate, '%Y-%m-%d')[0:6])
            oldutdate = datetime(2006, 8, 31, 0, 0)
            
            if (asList):
                retrdnoise = []
                for ext in dataset['SCI']:
                    # Descriptors must work for all AstroData Types so check
                    # if the original gain keyword exists to use for the
                    # look-up table
                    if ext.header.has_key(stdkeyDictGMOS['key_gmos_gainorig']):
                        headergain = \
                            ext.header[stdkeyDictGMOS['key_gmos_gainorig']]
                    else:
                        headergain = \
                            ext.header[stdkeyDictGMOS['key_gmos_gain']]
                    
                    ampname = ext.header[stdkeyDictGMOS['key_gmos_ampname']]
                    gmode = dataset.gainmode()
                    rmode = dataset.readspeedmode()
                    
                    rdnoisekey = (rmode, gmode, ampname)
                    
                    try:
                        if (obsutdate > oldutdate):
                            rdnoise = self.gmosampsRdnoise[rdnoisekey]
                        else:
                            rdnoise = \
                                self.gmosampsRdnoiseBefore20060831[rdnoisekey]
                    
                    except KeyError:
                        rdnoise = None
                    
                    retrdnoise.append(rdnoise)
            
            else:
                try:
                    if (dataset.nsciext() < 1):
                        # Descriptors must work for all AstroData Types so
                        # check if the original gain keyword exists to use for
                        # the look-up table
                        if hdu[1].header.has_key(stdkeyDictGMOS['key_gmos_gainorig']):
                            headergain = \
                                hdu[1].header[stdkeyDictGMOS['key_gmos_gainorig']]
                        else:
                            headergain = \
                                hdu[1].header[stdkeyDictGMOS['key_gmos_gain']]
                        
                        ampname = hdu[1].header[stdkeyDictGMOS['key_gmos_ampname']]
                        gmode = dataset.gainmode()
                        rmode = dataset.readspeedmode()
                        
                        rdnoisekey = (rmode, gmode, ampname)
                        
                        try:
                            if (obsutdate > oldutdate):
                                retrdnoise = self.gmosampsRdnoise[rdnoisekey]
                            else:
                                retrdnoise = \
                                    self.gmosampsRdnoiseBefore20060831[rdnoisekey]
                        except KeyError:
                            return None
                    else:
                        raise Exception('Please use asList=True to obtain a list')
                
                except Exception:
                    print traceback.format_exc()
        
        except KeyError:
            return None
        
        return retrdnoise
    
    gmosampsRdnoise = None
    gmosampsRdnoiseBefore20060831 = None
    
    def ronorig( self, dataset , **args):
        '''
        
        '''
        # Epic klugin' right here.
        # print 'GRD 692: called ronorig'
        temp = []
        try:
            for ext in dataset['SCI']:
                temp.append(ext.header['RONORIG'])
        except:
            temp = self.fetchValue('RDNOISE', dataset)
            
        #print 'GRD700:', repr(temp)
        return temp
    
    def readspeedmode(self, dataset, **args):
        """
        Return the read mode value for GMOS
        This is used in the gain descriptor for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the read mode
        """
        try:
            hdu = dataset.hdulist
            ampinteg = hdu[0].header[stdkeyDictGMOS['key_gmos_ampinteg']]
            
            if (ampinteg == 1000):
                retreadspeedmodestring = 'fast'
            else:
                retreadspeedmodestring = 'slow'
        
        except KeyError:
            return None
        
        return retreadspeedmodestring
    
    def satlevel(self, dataset, **args):
        """
        Return the satlevel value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the saturation level in the raw images (ADU)
        """
        retsaturationint = 65000
        
        return int(retsaturationint)
    
    def wdelta(self, dataset, asList=False, **args):
        """
        Return the wdelta value for GMOS
        If asList = True, a list is returned, where the number of array
        elements equals the number of pixel data extensions in the image.
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float or list (if asList = True)
        @returns: the dispersion value (angstroms/pixel)
        """
        try:
            if (asList):
                retwdelta = []
                for ext in dataset['SCI']:
                    wdelta = ext.header[stdkeyDictGMOS['key_gmos_wdelta']]
                    retwdelta.append(wdelta)
            
            else:
                try:
                    if (dataset.nsciext() < 1):
                        hdu = dataset.hdulist
                        retwdelta = \
                            hdu[1].header[stdkeyDictGMOS['key_gmos_wdelta']]
                    else:
                        raise Exception('Please use asList=True to obtain a list')
                
                except Exception:
                    print traceback.format_exc()
        
        except KeyError:
            return None
        
        return retwdelta
    
    def wrefpix(self, dataset, asList=False, **args):
        """
        Return the wrefpix value for GMOS
        If asList = True, a list is returned, where the number of array
        elements equals the number of pixel data extensions in the image.
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float or list (if asList = True)
        @returns: the reference pixel of the central wavelength
        """
        try:
            if (asList):
                retwrefpix = []
                for ext in dataset['SCI']:
                    wrefpix = ext.header[stdkeyDictGMOS['key_gmos_wrefpix']]
                    retwrefpix.append(wrefpix)
            
            else:
                try:
                    if (dataset.nsciext() < 1):
                        hdu = dataset.hdulist
                        retwrefpix = \
                            hdu[1].header[stdkeyDictGMOS['key_gmos_wrefpix']]
                    else:
                        raise Exception('Please use asList=True to obtain a list')
                
                except Exception:
                    print traceback.format_exc()
        
        except KeyError:
            return None
        
        return retwrefpix
    
    def xccdbin(self, dataset, **args):
        """
        Return the xccdbin value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the binning of the detector x-axis
        """
        try:
            hdu = dataset.hdulist
            ccdsum = hdu[1].header[stdkeyDictGMOS['key_gmos_ccdsum']]
            
            if ccdsum != None:
                retxccdbinint, yccdbin = ccdsum.split()
            else:
                return None
        
        except KeyError:
            return None
        
        return int(retxccdbinint)
    
    def yccdbin(self, dataset, **args):
        """
        Return the yccdbin value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the binning of the detector y-axis
        """
        try:
            hdu = dataset.hdulist
            ccdsum = hdu[1].header[stdkeyDictGMOS['key_gmos_ccdsum']]
            
            if ccdsum != None:
                xccdbin, retyccdbinint = ccdsum.split()
            else:
                return None
        
        except KeyError:
            return None
        
        return int(retyccdbinint)
