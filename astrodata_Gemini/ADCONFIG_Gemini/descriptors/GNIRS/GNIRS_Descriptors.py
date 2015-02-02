import math, re

from astrodata.utils import Errors
from astrodata.utils import Lookups
from astrodata.interface.Descriptors import DescriptorValue

from gempy.gemini import gemini_metadata_utils as gmu

from GNIRS_Keywords import GNIRS_KeyDict
from GEMINI_Descriptors import GEMINI_DescriptorCalc

# ------------------------------------------------------------------------------
class GNIRS_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    _update_stdkey_dict = GNIRS_KeyDict
    
    gnirsArrayDict = None
    gnirsConfigDict = None
    
    def __init__(self):
        self.gnirsArrayDict = Lookups.get_lookup_table(
            "Gemini/GNIRS/GNIRSArrayDict", "gnirsArrayDict")
        self.gnirsConfigDict = Lookups.get_lookup_table(
            "Gemini/GNIRS/GNIRSConfigDict", "gnirsConfigDict")
        GEMINI_DescriptorCalc.__init__(self)
    
    def disperser(self, dataset, stripID=False, pretty=False, **args):
        if pretty:
            stripID = True
        
        # GNIRS contains two dispersers - the grating and the prism. Get the
        # grating and the prism values using the appropriate descriptors
        grating = dataset.grating(stripID=stripID, pretty=pretty).as_pytype()
        prism = dataset.prism(stripID=stripID, pretty=pretty).as_pytype()
        
        if grating is None or prism is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Return the disperser string
        # If the "prism" is the "MIRror" then don't include it
        if prism.startswith('MIR'):
            ret_disperser = str(grating)
        else:
            ret_disperser = "%s&%s" % (grating, prism)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_disperser, name="disperser", ad=dataset)
        
        return ret_dv
    
    def focal_plane_mask(self, dataset, stripID=False, pretty=False, **args):
        # For GNIRS, the focal plane mask is the combination of the slit
        # mechanism and the decker mechanism. Get the slit and the decker
        # values using the appropriate descriptors
        slit = dataset.slit(stripID=stripID, pretty=pretty).as_pytype()
        decker = dataset.decker(stripID=stripID, pretty=pretty).as_pytype()
        
        if slit is None or decker is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Sometimes (2010 rebuild?) we see "Acquisition" and sometimes "Acq"
        # In both slit and decker. Make them all consistent.
        slit = slit.replace('Acquisition', 'Acq')
        decker = decker.replace('Acquisition', 'Acq')

        if pretty:
            # Disregard the decker if it's in long slit mode
            if "Long" in decker:
                focal_plane_mask = slit
            # Append XD to the slit name if the decker is in XD mode
            elif "XD" in decker:
                focal_plane_mask = "%s%s" % (slit, "XD")
            elif "IFU" in slit and "IFU" in decker:
                focal_plane_mask = "IFU"
            elif "Acq" in slit and "Acq" in decker:
                focal_plane_mask = "Acq"
            else:
                focal_plane_mask = "%s&%s" % (slit, decker)
        else:
            focal_plane_mask = "%s&%s" % (slit, decker)
        
        ret_focal_plane_mask = str(focal_plane_mask)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_focal_plane_mask, name="focal_plane_mask",
                                 ad=dataset)
        return ret_dv
    
    def gain(self, dataset, **args):
        # Determine the bias value keyword (biasvolt) from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_bias")
        
        # Get the value of the bias value keyword from the header of the PHU
        biasvolt = dataset.phu_get_key_value(keyword)
        
        if biasvolt is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
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
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_gain, name="gain", ad=dataset)
        
        return ret_dv
    
    gnirsArrayDict = None
    
    def grating(self, dataset, stripID=False, pretty=False, **args):
        """
        Note. A CC software change approx July 2010 changed the grating names
        to also include the camera, eg 32/mmSB_G5533 indicates the 32/mm
        grating with the Short Blue camera. This is unhelpful as if we wanted
        to know the camera, we'd call the camera descriptor. Thus, this
        descriptor function repairs the header values to only list the grating.
        """
        # Determine the grating keyword from the global keyword dictionary
        keyword = self.get_descriptor_key("key_grating")
        
        # Get the value of the grating keyword from the header of the PHU
        grating = dataset.phu_get_key_value(keyword)
        
        if grating is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # The format of the grating string is currently (2011) nnn/mmCAM_Gnnnn
        # nnn is a 2 or 3 digit number (lines per mm)
        # /mm is literally "/mm"
        # CAM is the camera: {L|S}{B|R}[{L|S}[X]}
        # _G is literally "_G"
        # nnnn is the 4 digit component ID.
        cre = re.compile("([\d/m]+)([A-Z]*)(_G)(\d+)")
        m = cre.match(grating)
        if m:
            parts = m.groups()
            ret_grating = "%s%s%s" % (parts[0], parts[2], parts[3])
        else:
            # If the regex didn't match, just pass through the raw value
            ret_grating = grating
        if stripID or pretty:
            ret_grating = gmu.removeComponentID(ret_grating)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_grating, name="grating", ad=dataset)
        
        return ret_dv
    
    def non_linear_level(self, dataset, **args):
        # Determine the bias value keyword (biasvolt) from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_bias")
        
        # Get the value of the bias value keyword from the header of the PHU
        biasvolt = dataset.phu_get_key_value(keyword)
        
        if biasvolt is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Get the saturation level using the appropriate descriptor
        saturation_level = dataset.saturation_level()
        
        if saturation_level is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Determine whether the dataset has been corrected for non-linearity
        if dataset.phu_get_key_value("NONLINCR"):
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
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_non_linear_level, name="non_linear_level",
                                 ad=dataset)
        return ret_dv
    
    gnirsArrayDict = None
    
    def pixel_scale(self, dataset, **args):
        # Determine the prism, decker and disperser keywords from the global
        # keyword dictionary
        keyword1 = self.get_descriptor_key("key_prism")
        keyword2 = self.get_descriptor_key("key_decker")
        keyword3 = self.get_descriptor_key("key_grating")
        
        # Get the value of the prism, decker and disperser keywords from the
        # header of the PHU 
        prism = dataset.phu_get_key_value(keyword1)
        decker = dataset.phu_get_key_value(keyword2)
        disperser = dataset.phu_get_key_value(keyword3)
        
        if prism is None or decker is None or disperser is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Get the camera using the appropriate descriptor
        camera = dataset.camera().as_pytype()
        
        if camera is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        pixel_scale_key = (prism, decker, disperser, camera)
        if pixel_scale_key in getattr(self, "gnirsConfigDict"):
            row = self.gnirsConfigDict[pixel_scale_key]
        else:
            raise Errors.TableKeyError()
        
        if float(row[2]):
            ret_pixel_scale = float(row[2])
        else:
            raise Errors.TableValueError()
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_pixel_scale, name="pixel_scale",
                                 ad=dataset)
        return ret_dv
    
    gnirsConfigDict = None
    
    def prism(self, dataset, stripID=False, pretty=False, **args):
        """
        Note. A CC software change approx July 2010 changed the prism names to
        also include the camera, eg 32/mmSB_G5533 indicates the 32/mm grating
        with the Short Blue camera. This is unhelpful as if we wanted to know
        the camera, we'd call the camera descriptor. Thus, this descriptor
        function repairs the header values to only list the prism.
        """
        # Determine the prism keyword from the global keyword dictionary
        keyword = self.get_descriptor_key("key_prism")
        
        # Get the value of the prism keyword from the header of the PHU.
        prism = dataset.phu_get_key_value(keyword)
        
        if prism is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # The format of the prism string is currently (2011) [CAM+]prism_Gnnnn
        # CAM is the camera: {L|S}{B|R}[{L|S}[X]}
        # + is a literal "+"
        # prism is the actual prism name
        # nnnn is the 4 digit component ID.
        # The change from the old style is the camer prefix, which we drop here
        cre = re.compile("([LBSR]*\+)*([A-Z]*_G\d+)")
        m = cre.match(prism)
        if m:
            ret_prism = m.group(2)

        if stripID or pretty:
                ret_prism = gmu.removeComponentID(ret_prism)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_prism, name="prism", ad=dataset)
        
        return ret_dv
    
    def read_mode(self, dataset, **args):
        # Determine the number of non-destructive read pairs (lnrs) and the
        # number of digital averages (ndavgs) keywords from the global keyword
        # dictionary
        keyword1 = self.get_descriptor_key("key_lnrs")
        keyword2 = self.get_descriptor_key("key_ndavgs")
        
        # Get the value of the number of non-destructive read pairs and the
        # number of digital averages keywords from the header of the PHU
        lnrs = dataset.phu_get_key_value(keyword1)
        ndavgs = dataset.phu_get_key_value(keyword2)
        
        if lnrs is None or ndavgs is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        if lnrs == 32 and ndavgs == 16:
            read_mode = "Very Faint Objects"
        elif lnrs == 16 and ndavgs == 16:
            read_mode = "Faint Objects"
        elif lnrs == 1 and ndavgs == 16:
            read_mode = "Bright Objects"
        elif lnrs == 1 and ndavgs == 1:
            read_mode = "Very Bright Objects"
        else:
            read_mode = "Invalid"
        
        ret_read_mode = str(read_mode)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_read_mode, name="read_mode", ad=dataset)
        
        return ret_dv
    
    def read_noise(self, dataset, **args):
        # Determine the bias value (biasvolt), the number of non-destructive
        # read pairs (lnrs) and the number of digital averages (ndavgs)
        # keywords from the global keyword dictionary
        keyword1 = self.get_descriptor_key("key_bias")
        keyword2 = self.get_descriptor_key("key_lnrs")
        keyword3 = self.get_descriptor_key("key_ndavgs")
        
        # Get the value of the bias value, the number of non-destructive read
        # pairs and the number of digital averages keywords from the header of
        # the PHU
        biasvolt = dataset.phu_get_key_value(keyword1)
        lnrs = dataset.phu_get_key_value(keyword2)
        ndavgs = dataset.phu_get_key_value(keyword3)
        
        if biasvolt is None or lnrs is None or ndavgs is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Get the number of coadds using the appropriate descriptor
        coadds = dataset.coadds()
        
        if coadds is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
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
        
        ret_read_noise = float((read_noise * math.sqrt(coadds)) /
                               (math.sqrt(lnrs) * math.sqrt(ndavgs)))
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_read_noise, name="read_noise", ad=dataset)
        
        return ret_dv
    
    gnirsArrayDict = None
    
    def saturation_level(self, dataset, **args):
        # Determine the bias value (biasvolt) keyword from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_bias")
        
        # Get the value of the bias value keyword from the header of the PHU
        biasvolt = dataset.phu_get_key_value(keyword)
        
        if biasvolt is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Get the number of coadds using the appropriate descriptor
        coadds = dataset.coadds()
        
        if coadds is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
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
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_saturation_level, name="saturation_level",
                                 ad=dataset)
        return ret_dv
    
    gnirsArrayDict = None
    
    def slit(self, dataset, stripID=False, pretty=False, **args):
        """
        Note that in GNIRS all the slits are machined into one physical piece
        of metal, which is on a slide - the mechanism simply slides the slide
        along to put the right slit in the beam. Thus all the slits have the
        same component ID as they're they same physical component.

        Note that in the ~2010 rebuild, the slit names were changed to remove
        the space - ie "1.00 arcsec" -> "1.00arcsec"
        So here, we remove the space all the time for consistency.
        """
        # Determine the slit keyword from the global keyword dictionary
        keyword = self.get_descriptor_key("key_slit")
        
        # Get the value of the slit keyword from the header of the PHU
        slit = dataset.phu_get_key_value(keyword)
        
        if slit is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info

        slit = slit.replace(' ', '')
        
        if stripID or pretty:
            ret_slit = gmu.removeComponentID(slit)
        else:
            ret_slit = str(slit)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_slit, name="slit", ad=dataset)
        
        return ret_dv
    
    def well_depth_setting(self, dataset, **args):
        # Determine the bias value (biasvolt) keyword from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_bias")
        
        # Get the value of the bias value keyword from the header of the PHU
        biasvolt = dataset.phu_get_key_value(keyword)
        
        if biasvolt is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        if abs(0.3 - abs(biasvolt)) < 0.1:
            well_depth_setting = "Shallow"
        elif abs(0.6 - abs(biasvolt)) < 0.1:
            well_depth_setting = "Deep"
        else:
            well_depth_setting = "Invalid"
        
        # Return the well depth setting string
        ret_well_depth_setting = str(well_depth_setting)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_well_depth_setting,
                                 name="well_depth_setting", ad=dataset)
        return ret_dv
