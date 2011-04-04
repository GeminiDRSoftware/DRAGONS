import math, re

from astrodata import Descriptors
from astrodata import Errors
from astrodata import Lookups
from astrodata.Calculator import Calculator
from gempy import string

from StandardDescriptorKeyDict import globalStdkeyDict
from StandardGNIRSKeyDict import stdkeyDictGNIRS
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class GNIRS_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    globalStdkeyDict.update(stdkeyDictGNIRS)
    
    gnirsArrayDict = None
    gnirsConfigDict = None
    
    def __init__(self):
        self.gnirsArrayDict = \
            Lookups.getLookupTable('Gemini/GNIRS/GNIRSArrayDict',
                                   'gnirsArrayDict')
        self.gnirsConfigDict = \
            Lookups.getLookupTable('Gemini/GNIRS/GNIRSConfigDict',
                                   'gnirsConfigDict')
    
    def disperser(self, dataset, stripID=False, pretty=False, **args):
        if pretty:
            stripID = True
        # GNIRS contains two dispersers - the grating and the prism. Get the
        # grating and the prism values using the appropriate descriptors
        grating = dataset.grating(stripID=stripID, pretty=pretty).asPytype()
        prism = dataset.prism(stripID=stripID, pretty=pretty).asPytype()
        if grating is None or prism is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        if stripID:
            if pretty and prism.startswith('MIR'):
                # Return the stripped and pretty disperser string. If the
                # prism is a mirror, don't list it in the pretty disperser. 
                disperser = string.removeComponentID(grating)
            else:
                # Return the stripped disperser string
                disperser = '%s&%s' % (string.removeComponentID(grating), \
                    string.removeComponentID(prism))
        else:
            # Return the disperser string
            disperser = '%s&%s' % (grating, prism)
        
        ret_disperser = str(disperser)
        
        return ret_disperser
    
    def focal_plane_mask(self, dataset, stripID=False, pretty=False, **args):
        # For GNIRS, the focal plane mask is the combination of the slit
        # mechanism and the decker mechanism. Get the slit and the decker
        # values using the appropriate descriptors
        slit = dataset.slit(stripID=stripID, pretty=pretty)
        decker = dataset.decker(stripID=stripID, pretty=pretty).asPytype()
        if slit is None or decker is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        if pretty:
            # Disregard the decker if it's in long slit mode
            if 'Long' in decker:
                focal_plane_mask = slit
            # Append XD to the slit name if the decker is in XD mode
            elif 'XD' in decker:
                focal_plane_mask = '%s%s' % (slit, 'XD')
            else:
                focal_plane_mask = '%s&%s' % (slit, decker)
        else:
            focal_plane_mask = '%s&%s' % (slit, decker)
        ret_focal_plane_mask = str(focal_plane_mask)
        
        return ret_focal_plane_mask
    
    def gain(self, dataset, **args):
        # Get the bias value (biasvolt) from the header of the PHU. The bias
        # keyword is defined in the local key dictionary (stdkeyDictGNIRS) but
        # is read from the updated global key dictionary (globalStdkeyDict)
        biasvolt = dataset.phuGetKeyValue(globalStdkeyDict['key_bias'])
        if biasvolt is None:
            # The phuGetKeyValue() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        bias_values = self.gnirsArrayDict.keys()
        count = 0
        for bias in bias_values:
            if abs(float(bias) - abs(biasvolt)) < 0.1:
                count += 1
                if float(self.gnirsArrayDict[bias][2]):
                    ret_gain = float(self.gnirsArrayDict[bias][2])
                else:
                    Errors.TableValueError()
        if count == 0:
            Errors.TableKeyError()
        
        return ret_gain
    
    gnirsArrayDict = None
    
    def grating(self, dataset, stripID=False, pretty=False, **args):
        """
        Note. A CC software change approx July 2010 changed the grating names
        to also include the camera, eg 32/mmSB_G5533 indicates the 32/mm
        grating with the Short Blue camera. This is unhelpful as if we wanted
        to know the camera, we'd call the camera descriptor. Thus, this
        descriptor function repairs the header values to only list the grating.
        """
        # Get the grating value from the header of the PHU.
        grating = dataset.phuGetKeyValue(globalStdkeyDict['key_grating'])
        if grating is None:
            # The phuGetKeyValue() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        # The format of the grating string is currently (2011) nnn/mmCAM_Gnnnn
        # nnn is a 2 or 3 digit number (lines per mm)
        # /mm is literally '/mm'
        # CAM is the camera: {L|S}{B|R}[{L|S}[X]}
        # _G is literally '_G'
        # nnnn is the 4 digit component ID.
        cre = re.compile('([\d/m]+)([A-Z]*)(_G)(\d+)')
        m = cre.match(grating)
        if m:
            parts = m.groups()
            ret_grating = '%s%s%s' % (parts[0], parts[2], parts[3])
            if stripID or pretty:
                ret_grating = string.removeComponentID(ret_grating)
        
        return ret_grating
    
    def non_linear_level(self, dataset, **args):
        # Get the bias value (biasvolt) from the header of the PHU. The bias
        # keyword is defined in the local key dictionary (stdkeyDictGNIRS) but
        # is read from the updated global key dictionary (globalStdkeyDict)
        biasvolt = dataset.phuGetKeyValue(globalStdkeyDict['key_bias'])
        if biasvolt is None:
            # The phuGetKeyValue() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        # Get the saturation level using the appropriate descriptor
        saturation_level = dataset.saturation_level()
        if saturation_level is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        # Determine whether the dataset has been corrected for non-linearity
        if dataset.phuGetKeyValue('NONLINCR'):
            corrected = True
        else:
            corrected = False
        # The array is non-linear at some fraction of the saturation level.
        # Get this fraction from the lookup table
        bias_values = self.gnirsArrayDict.keys()
        count = 0
        for bias in bias_values:
            if abs(float(bias) - abs(biasvolt)) < 0.1:
                count += 1
                row = self.gnirsArrayDict[bias]
                if corrected:
                    # Use row[4] if correcting for non-linearity
                    if float(row[4]):
                        linearlimit = float(row[4])
                    else:
                        Errors.TableValueError()
                else:
                    # Use row[8] if not correcting for non-linearity
                    if float(row[8]):
                        linearlimit = float(row[8])
                    else:
                        Errors.TableValueError()
        if count == 0:
            Errors.TableKeyError()
        # Return the saturation level integer
        ret_non_linear_level = int(saturation_level * linearlimit)
        
        return ret_non_linear_level
    
    gnirsArrayDict = None
    
    def pixel_scale(self, dataset, **args):
        # Get the prism, decker and disperser from the header of the PHU.
        prism = dataset.phuGetKeyValue(globalStdkeyDict['key_prism'])
        decker = dataset.phuGetKeyValue(globalStdkeyDict['key_decker'])
        disperser = dataset.phuGetKeyValue(globalStdkeyDict['key_grating'])
        if prism is None or decker is None or disperser is None:
            # The phuGetKeyValue() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        # Get the camera using the appropriate descriptor
        camera = dataset.camera().asPytype()
        if camera is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        pixel_scale_key = (prism, decker, disperser, camera)
        if pixel_scale_key in getattr(self, 'gnirsConfigDict'):
            row = self.gnirsConfigDict[pixel_scale_key]
        else:
            raise Errors.TableKeyError()
        if float(row[2]):
            ret_pixel_scale = float(row[2])
        else:
            raise Errors.TableValueError()
        
        return ret_pixel_scale
    
    gnirsConfigDict = None
    
    def prism(self, dataset, stripID=False, pretty=False, **args):
        """
        Note. A CC software change approx July 2010 changed the prism names to
        also include the camera, eg 32/mmSB_G5533 indicates the 32/mm grating
        with the Short Blue camera. This is unhelpful as if we wanted to know
        the camera, we'd call the camera descriptor. Thus, this descriptor
        function repairs the header values to only list the prism.
        """
        # Get the prism value from the header of the PHU.
        prism = dataset.phuGetKeyValue(globalStdkeyDict['key_prism'])
        if prism is None:
            # The phuGetKeyValue() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        # The format of the prism string is currently (2011) [CAM+]prism_Gnnnn
        # CAM is the camera: {L|S}{B|R}[{L|S}[X]}
        # + is a literal '+'
        # prism is the actual prism name
        # nnnn is the 4 digit component ID.
        cre = re.compile('([LBSR]*\+)*([A-Z]*)(_G)(\d+)')
        m = cre.match(prism)
        if m:
            parts = m.groups()
            ret_prism = '%s%s%s' % (parts[1], parts[2], parts[3])
            if stripID or pretty:
                ret_prism = string.removeComponentID(ret_prism)
        
        return ret_prism
    
    def read_mode(self, dataset, **args):
        # Get the number of non-destructive read pairs (lnrs) and the number
        # of digital averages (ndavgs) from the header of the PHU. The lnrs and
        # ndavgs keywords are defined in the local key dictionary
        # (stdkeyDictGNIRS) but are read from the updated global key dictionary
        # (globalStdkeyDict)
        lnrs = dataset.phuGetKeyValue(globalStdkeyDict['key_lnrs'])
        ndavgs = dataset.phuGetKeyValue(globalStdkeyDict['key_ndavgs'])
        if lnrs is None or ndavgs is None:
            # The phuGetKeyValue() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        if lnrs == 32 and ndavgs == 16:
            read_mode = 'Very Faint Objects'
        elif lnrs == 16 and ndavgs == 16:
            read_mode = 'Faint Objects'
        elif lnrs == 1 and ndavgs == 16:
            read_mode = 'Bright Objects'
        elif lnrs == 1 and ndavgs == 1:
            read_mode = 'Very Bright Objects'
        else:
            read_mode = 'Invalid'
        ret_read_mode = str(read_mode)
        
        return ret_read_mode
    
    def read_noise(self, dataset, **args):
        # Get the bias value (biasvolt), the number of non-destructive read
        # pairs (lnrs) and the number of digital averages (ndavgs) from the
        # header of the PHU. The biasvolt, lnrs and ndavgs keywords are
        # defined in the local key dictionary (stdkeyDictGNIRS) but are read
        # from the updated global key dictionary (globalStdkeyDict)
        biasvolt = dataset.phuGetKeyValue(globalStdkeyDict['key_bias'])
        lnrs = dataset.phuGetKeyValue(globalStdkeyDict['key_lnrs'])
        ndavgs = dataset.phuGetKeyValue(globalStdkeyDict['key_ndavgs'])
        if biasvolt is None or lnrs is None or ndavgs is None:
            # The phuGetKeyValue() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        # Get the number of coadds using the appropriate descriptor
        coadds = dataset.coadds()
        if coadds is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        bias_values = self.gnirsArrayDict.keys()
        count = 0
        for bias in bias_values:
            if abs(float(bias) - abs(biasvolt)) < 0.1:
                count += 1
                if float(self.gnirsArrayDict[bias][1]):
                    read_noise = float(self.gnirsArrayDict[bias][1])
                else:
                    Errors.TableValueError()
        if count == 0:
            Errors.TableKeyError()
        
        ret_read_noise = float((read_noise * math.sqrt(coadds)) \
                / (math.sqrt(lnrs) * math.sqrt(ndavgs)))
        
        return ret_read_noise
    
    gnirsArrayDict = None
    
    def saturation_level(self, dataset, **args):
        # Get the bias value (biasvolt) from the header of the PHU. The
        # biasvolt keyword is defined in the local key dictionary
        # (stdkeyDictGNIRS) but is read from the updated global key dictionary
        # (globalStdkeyDict)
        biasvolt = dataset.phuGetKeyValue(globalStdkeyDict['key_bias'])
        if biasvolt is None:
            # The phuGetKeyValue() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        # Get the number of coadds using the appropriate descriptor
        coadds = dataset.coadds()
        if coadds is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        bias_values = self.gnirsArrayDict.keys()
        count = 0
        for bias in bias_values:
            if abs(float(bias) - abs(biasvolt)) < 0.1:
                count += 1
                if float(self.gnirsArrayDict[bias][3]):
                    well = float(self.gnirsArrayDict[bias][3])
                else:
                    Errors.TableValueError()
        if count == 0:
            Errors.TableKeyError()
        
        ret_saturation_level = int(well * coadds)
        
        return ret_saturation_level
    
    gnirsArrayDict = None
    
    def slit(self, dataset, stripID=False, pretty=False, **args):
        """
        Note that in GNIRS all the slits are machined into one physical piece
        of metal, which is on a slide - the mechanism simply slides the slide
        along to put the right slit in the beam. Thus all the slits have the
        same componenet ID as they're they same physical compononet.
        """
        # Get the slit value from the header of the PHU.
        slit = dataset.phuGetKeyValue(globalStdkeyDict['key_slit'])
        if slit is None:
            # The phuGetKeyValue() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        if stripID or pretty:
            ret_slit = string.removeComponentID(slit)
        else:
            ret_slit = str(slit)
        
        return ret_slit
    
    def well_depth_setting(self, dataset, **args):
        # Get the bias value (biasvolt) from the header of the PHU. The
        # biasvolt keyword is defined in the local key dictionary
        # (stdkeyDictGNIRS) but is read from the updated global key dictionary
        # (globalStdkeyDict)
        biasvolt = dataset.phuGetKeyValue(globalStdkeyDict['key_bias'])
        if biasvolt is None:
            # The phuGetKeyValue() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        if abs(biasvolt + 0.3) < 0.1:
            well_depth_setting = 'Deep'
        elif abs(biasvolt + 0.6) < 0.1:
            well_depth_setting = 'Shallow'
        else:
            well_depth_setting = 'Invalid'
        # Return the well depth setting string
        ret_well_depth_setting = str(well_depth_setting)
        
        return well_depth_setting
