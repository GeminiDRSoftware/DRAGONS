import os
import pyfits
from xml.dom.minidom import parse
#------------------------------------------------------------------------------ 
import astrodata
from astrodata import AstroData
import ConfigSpace
# double import import Descriptors
import gdpgutil
from ReductionObjectRequests import CalibrationRequest
#------------------------------------------------------------------------------ 
from astrodata.adutils import gemLog
from Errors import CalibrationDefinitionLibraryError as CDLExcept
from Errors import AstroDataError
from Errors import ExistError
from Errors import DescriptorTypeError
from Errors import FatalDeprecation               
class CalibrationDefinitionLibrary(object):
    '''
    This class deals with obtaining request data from XML calibration files and generating 
    the corresponding request.  
    '''  
    log=None  
    
    def __init__(self):
        '''
        Goes into ConfigSpace and gets all the file URIs to create a XML index.
        '''
        # raise FatalDeprecation("CalibrationDefinitionLibrary NOT IN USE!")
        
        self.xmlIndex = {}
        #self.update_xml_index()
        # instantiate the logger object and put into the global variable 
        self.log = gemLog.getGeminiLog()
            
        
    def OBSOLETE_update_xml_index(self):
        '''
        Re-updates the xml index, could be useful if this becomes long running and there are changes to
        the xml files, etc.
        '''
        
        self.xmlIndex = {}
        try:
            for dpath, dnames, files in ConfigSpace.config_walk( "xmlcalibrations" ):
                for file in files:
                    self.xmlIndex.update( {file:os.path.join(str(dpath), file)} )
            #print "CDL30", self.xmlIndex
        except:
            raise "Could not load XML Index."
       
    def get_cal_req(self, inputs, caltype, write_input = False):
        """
        For each input finds astrodata type to find corresponding xml file,
        loads the file.         
        
        @param inputs: list of input fits AstroData instances
        @type inputs: list
        
        @param caltype: Calibration, ie bias, flat, dark, etc.
        @type caltype: string
        
        @return: Returns a list of Calibration Request Events.
        @rtype: list
        """        
        reqEvents = []
        
        for inp in inputs:
            cr = CalibrationRequest()
            # print "CDL56:", repr(inp), inp.filename, str(inp)
            # @@REVIEW: write_input is a bad name... you write outputs!
            if write_input == True:
                if (False):
                # don't write files... we will use headers
                    if os.path.exists(inp.filename):
                        # then asked to write something already on disk and we
                        # don't want to blindly clobber... throw informative error
                        msg = "Overwriting %s with in-memory version " + \
                                     "to ensure a current version of the dataset " + \
                                     "is available " + \
                                     "to Calibration Service." 
                        msg = msg % inp.filename
                        self.log.warning(msg)
                    else:
                        self.log.status("Writing in-memory AstroData instance " +
                                   "to new disk file (%s) to ensure availability " +
                                   "to Calibration Service." % inp.filename)
                try:
                    inp.write(clobber = True)
                except AstroDataError("Mode is readonly"):
                    self.log.warning("Skipped writing dataset, as it was "
                                "readonly input. This write "
                                "is done to ensure the file in memory is on disk "
                                "as the calibration system inspects the file itself. "
                                "As the file is protected as readonly, the system will "
                                "assume it is unchanged since loading.")
            
            cr.filename = inp.filename
            cr.ad = inp
            cr.caltype = caltype
            # @@NOTE: should use IDFactory, not data_label which HAPPENS to be the id
            cr.datalabel = inp.data_label().for_db()
            
            ad = inp # saves me time, as I cut/pasted the below from a test script

            # Old version of the descriptor dictionary: this fails badly
            # for any data type that doesn't have all these descriptors
            # defined
            """
            #print "CDL99:", str(ad.ut_datetime())
            cr.descriptors =  {'instrument':ad.instrument().for_db(),
                         'observation_type': ad.observation_type().for_db(),
                         'data_label': ad.data_label().for_db(),
                         'detector_x_bin':ad.detector_x_bin().for_db(),
                         'detector_y_bin':ad.detector_y_bin().for_db(),
                         'read_speed_setting':ad.read_speed_setting().for_db(),
                         'gain_setting':ad.gain_setting().for_db(),
                         'amp_read_area':ad.amp_read_area().for_db(),
                         #'ut_datetime':repr(ad.ut_datetime().for_db()),
                         'ut_datetime':ad.ut_datetime().for_db(),
                         'exposure_time':ad.exposure_time().for_db(),
                         'object': ad.object().for_db(),
                         'filter_name':ad.filter_name().for_db(),
                         'focal_plane_mask':ad.focal_plane_mask().for_db(),
                         }
            """
            # List of all possible needed descriptors
            descriptor_list = ['amp_read_area',
                               'central_wavelength',
                               'data_label',
                               'detector_x_bin',
                               'detector_y_bin',
                               'disperser',
                               'exposure_time',
                               'filter_name',
                               'focal_plane_mask',
                               'gain_setting',
                               'instrument',
                               'nod_count',
                               'nod_pixels',
                               'object',
                               "observation_class",
                               'observation_type',
                               'program_id',
                               'read_speed_setting',
                               'ut_datetime',
                               'detector_roi_setting'
                               ]
            options = {'central_wavelength':'asMicrometers=True'}

            # Check that each descriptor works and returns a 
            # sensible value before adding it to the dictionary
            desc_dict = {}
            for desc_name in descriptor_list:
                if options.has_key(desc_name):
                    opt = options[desc_name]
                else:
                    opt = ''
                try:
                    exec_cmd = 'dv = ad.%s(%s)' % (desc_name,opt)
                    exec(exec_cmd)
                    # print "cdl161:"+exec_cmd+"=="+str(dv)
                except (ExistError,KeyError,DescriptorTypeError):
                    continue
                if dv is not None:
                    desc_dict[desc_name] = dv.for_db()
                else:
                    desc_dict[desc_name] = None
                
            cr.descriptors = desc_dict
            cr.types = ad.types
            
            reqEvents.append(cr)
            
        # Goes to reduction context object to add to queue
        return reqEvents
    
    
    def parse_query(self, xml_dom_query_node, caltype, inputf ):
        '''
        Parses a query from XML Calibration File and returns a Calibration
        request event with the corresponding information. Unfinished: priority 
        parsing value
        
        @param xml_dom_query_node: a query XML Dom Node; ie <DOM Element: query at 0x921392c>
        @type xml_dom_query_node: Dom Element
        
        @param caltype: Calibration, ie bias, flat, dark, etc.
        @type caltype: string
        
        @param input: an input fits URI
        @type input: string
        
        @return: Returns a Calibration Request Event.
        @rtype: CalibrationRequestEvent
        '''
        import Descriptors # bad to load on import, import mess
        calReqEvent = CalibrationRequest()
        calReqEvent.caltype = caltype
        query = xml_dom_query_node
        
        if not query.hasAttribute("id"):
            raise "Improperly formed. QUERY needs an id, for example 'bias'."
        
        tempcal = str(query.getAttribute("id"))
        
        if( tempcal != caltype ):
            raise "The id in the query does not match the caltype '"+tempcal+"' '"+str(caltype)+"'."
        
        if type( inputf ) == AstroData:
            ad = inputf
        elif type( inputf ) == str:
            ad = AstroData( inputf )
        else:
            raise RuntimeError("Bad Argument: Wrong Type, '%(val)s' '%(typ)s'." %{'val':str(inputf),'typ':str(type(inputf))})
        desc = Descriptors.get_calculator( ad )
        #===============================================================
        # IDENTIFIERS
        #===============================================================
        identifiers = query.getElementsByTagName( "identifier" )
        if len( identifiers ) > 0:
            identifiers = identifiers[0]
        else:
            raise "Improperly formed. XML calibration has no identifiers."
        
        
        for child in identifiers.getElementsByTagName( "property" ):
            #creates dictionary object with multiple values    
            temp = self.parse_property(child, desc, ad)  
            #print "CDL112:", temp         
            calReqEvent.identifiers.update( temp ) 
        
        #===============================================================
        # CRITERIA
        #===============================================================
        criteria = query.getElementsByTagName( "criteria" )
        if len( criteria ) > 0:
            criteria = criteria[0]
        else:
            raise "Improperly formed. XML calibration has no criteria" 
        
        #print 'Locating a %s for %s.' %(str(caltype), str(ad.filename))
        #print 'Using the following criteria:'
        
        for child in criteria.getElementsByTagName( "property" ):
            crit = self.parse_property( child, desc, ad )
            calReqEvent.criteria.update( crit )      
            # print self.str_property( crit )
        
        #===============================================================
        # PRIORITIES
        #===============================================================
        priorities = query.getElementsByTagName( "priorities" )
        if len( priorities ) > 0:
            priorities = priorities[0]
        else:
            raise "Improperly formed. XML calibration has no priorities"
        
        for child in priorities.getElementsByTagName( "property" ):
            calReqEvent.priorities.update( self.parse_property(child, desc, ad) )
        
        
        calReqEvent.filename = inputf
        return calReqEvent
    
                  
    def parse_property( self, property_node, desc, ad ):
        '''
        Parses a xmldom property, returning a {key:(extension,elemType,value)}.
        
        @param property_node: xmlDom Element, that should be a 'property'. Consult the xml calibration file
        definitions for more information.
        @type property_node:  Dom Element
        
        @param desc: Descriptor for the type ad.
        @type desc: Calculator
        
        @param ad: An Astrodata instance for the input file.
        @type ad: Astrodata instance
        
        @return: {key:(extension,elemType,value)}, based of calibration xml attributes.
        @rtype: dict
        '''
        #--KEY-------------------------------------------------------------------------- 
        if not property_node.hasAttribute( "key" ):
            raise "Improperly formed XML calibration. A 'key' attribute is missing in one " + \
                "of the 'property' elements."
        key = property_node.getAttribute( "key" )
        
        #--EXT-------------------------------------------------------------------------- 
        # This might become obsolete
        extension = "PHU"
        if property_node.hasAttribute( "extension" ):
            extension = property_node.getAttribute( "extension" )
        
        #--TYP-------------------------------------------------------------------------- 
        # This might become obsolete
        elemType = "string"
        if property_node.hasAttribute( "type" ):
            elemType = property_node.getAttribute( "type" )
        
        #--VAL-------------------------------------------------------------------------- 
        if property_node.hasAttribute( "value" ) and property_node.getAttribute( "value" ) != "":
            value = property_node.getAttribute( "value" )
        else:
            value = desc.fetch_value( str(key), ad )
        
        return {key:(extension,elemType,value)}
    
    
    def generate_cal_index( self, caltype ):
        '''
        Generate a xml URI index for each caltype. This could seem kind of inefficient, but 
        this is used to take advantage of the generalized utility function pick_config.
        
        @param caltype: The calibration needed to generate the index.
        @type caltype: string    
        
        @return: Returns a dictionary of the form: {DataClassification.name:xmlFile}.
        For example: {'GMOS':'GMOS-bias.xml','NIRI':'NIRI-bias.xml'}. 
        @rtype: dict
        '''
        calIndex = {}
        raise FatalDeprecation("CalibrationDefinitionLibrary NOT IN USE!")
        
        for calFile in self.xmlIndex.keys():
            # calFile will look like 'GMOS-bias.xml', for split reference below.
            temp = calFile.split( "-" )
            adType = temp[0]
            tempCal = temp[1].split( '.' )[0]
            if tempCal == caltype:
                calIndex.update( {adType:calFile} )
        print "CDL213:", calIndex
        return calIndex

    
    def str_property(self, prop):
        '''
        A cleaner way to print properties out.
        
        @param prop: A property as defined by the xml schema.
        @type prop: dict
        
        @return: A clean, one line property.
        @rtype: str 
        '''
        retStr = ''
        for key in prop.keys():
            vals = prop[key]
            retStr += '${GREEN}%s:${NORMAL}\t%s${NORMAL}' %(str(key),str(vals[2])) 
        
        return retStr
        


    def __str__(self):
        return str(self.xmlIndex)
