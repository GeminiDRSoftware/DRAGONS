import os
import pyfits
from xml.dom.minidom import parse

from AstroData import AstroData
import ConfigSpace
import Descriptors
import gdpgutil
from ReductionObjectRequests import CalibrationRequest
#------------------------------------------------------------------------------ 
        
class CalibrationDefinitionLibrary( object ):
    '''
    This class deals with obtaining request data from XML calibration files and generating 
    the corresponding request.    
    '''    
    
    def __init__( self ):
        '''
        Goes into ConfigSpace and gets all the file URIs to create a XML index.
        '''
        self.xmlIndex = {}
        self.updateXmlIndex()
        
    def updateXmlIndex(self):
        '''
        Re-updates the xml index, could be useful if this becomes long running and there are changes to
        the xml files, etc.
        '''
        self.xmlIndex = {}
        try:
            for dpath, dnames, files in ConfigSpace.configWalk( "xmlcalibrations" ):
                for file in files:
                    self.xmlIndex.update( {file:os.path.join(str(dpath), file)} )
            #print "CDL30", self.xmlIndex
        except:
            raise "Could not load XML Index."
       
    def getCalReq(self, inputs, caltype):
        """
        For each input finds astrodata type to find corresponding xml file,
        loads the file.         
        
        @param inputs: list of input fits URIs
        @type inputs: list
        
        @param caltype: Calibration, ie bias, flat, dark, etc.
        @type caltype: string
        
        @return: Returns a list of Calibration Request Events.
        @rtype: list
        """
        reqEvents = []
        for input in inputs:
            calIndex = self.generateCalIndex( caltype)
            retDict = gdpgutil.pickConfig( input, self.generateCalIndex(caltype) )
            key = retDict.keys()[0]
            filename = calIndex[key]            
            
            try:
                calXMLURI = self.xmlIndex[filename]
                calXMLFile = open( calXMLURI, 'r' )
                xmlDom = parse( calXMLFile )
            except:
                raise "Error opening '%s'" %calXMLURI
            finally:
                calXMLFile.close()
                
            # childNodes is the <query> tag(s)           
            calReqEvent = self.parseQuery( xmlDom.childNodes[0], caltype, input )            
            reqEvents.append(calReqEvent)
        # Goes to reduction context object to add to queue
        return reqEvents
    
    
    def parseQuery(self, xmlDomQueryNode, caltype, input ):
        '''
        Parses a query from XML Calibration File and returns a Calibration
        request event with the corresponding information. Unfinished: priority 
        parsing value
        
        @param xmlDomQueryNode: a query XML Dom Node; ie <DOM Element: query at 0x921392c>
        @type xmlDomQueryNode: Dom Element
        
        @param caltype: Calibration, ie bias, flat, dark, etc.
        @type caltype: string
        
        @param input: an input fits URI
        @type input: string
        
        @return: Returns a Calibration Request Event.
        @rtype: CalibrationRequestEvent
        '''
       
        calReqEvent = CalibrationRequest()
        calReqEvent.caltype = caltype
        query = xmlDomQueryNode
        
        if not query.hasAttribute("id"):
            raise "Improperly formed. QUERY needs an id, for example 'bias'."
        
        tempcal = str(query.getAttribute("id"))
        
        if( tempcal != caltype ):
            raise "The id in the query does not match the caltype '"+tempcal+"' '"+str(caltype)+"'."
        
        ad = AstroData( input )
        desc = Descriptors.getCalculator( ad )
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
            temp = self.parseProperty(child, desc, ad)  
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
        
        for child in criteria.getElementsByTagName( "property" ):
            calReqEvent.criteria.update( self.parseProperty(child, desc, ad) )
        
        #===============================================================
        # PRIORITIES
        #===============================================================
        priorities = query.getElementsByTagName( "priorities" )
        if len( priorities ) > 0:
            priorities = priorities[0]
        else:
            raise "Improperly formed. XML calibration has no priorities"
        
        for child in priorities.getElementsByTagName( "property" ):
            calReqEvent.priorities.update( self.parseProperty(child, desc, ad) )
        
        
        calReqEvent.filename = input                           
        return calReqEvent
    
                  
    def parseProperty( self, propertyNode, desc, ad ):
        '''
        Parses a xmldom property, returning a {key:(extension,elemType,value)}.
        
        @param propertyNode: xmlDom Element, that should be a 'property'. Consult the xml calibration file
        definitions for more information.
        @type propertyNode:  Dom Element
        
        @param desc: Descriptor for the type ad.
        @type desc: Calculator
        
        @param ad: An Astrodata instance for the input file.
        @type ad: Astrodata instance
        
        @return: {key:(extension,elemType,value)}, based of calibration xml attributes.
        @rtype: dict
        '''
        #--KEY-------------------------------------------------------------------------- 
        if not propertyNode.hasAttribute( "key" ):
            raise "Improperly formed XML calibration. A 'key' attribute is missing in one " + \
                "of the 'property' elements."
        key = propertyNode.getAttribute( "key" )
        
        #--EXT-------------------------------------------------------------------------- 
        # This might become obsolete
        extension = "PHU"
        if propertyNode.hasAttribute( "extension" ):
            extension = propertyNode.getAttribute( "extension" )
        
        #--TYP-------------------------------------------------------------------------- 
        # This might become obsolete
        elemType = "string"
        if propertyNode.hasAttribute( "type" ):
            elemType = propertyNode.getAttribute( "type" )
        
        #--VAL-------------------------------------------------------------------------- 
        if propertyNode.hasAttribute( "value" ) and propertyNode.getAttribute( "value" ) != "":
            value = propertyNode.getAttribute( "value" )
        else:
            value = desc.fetchValue( str(key), ad )
        
        return {key:(extension,elemType,value)}
    
    
    def generateCalIndex( self, caltype ):
        '''
        Generate a xml URI index for each caltype. This could seem kind of inefficient, but 
        this is used to take advantage of the generalized utility function pickConfig.
        
        @param caltype: The calibration needed to generate the index.
        @type caltype: string    
        
        @return: Returns a dictionary of the form: {DataClassification.name:xmlFile}.
        For example: {'GMOS':'GMOS-bias.xml','NIRI':'NIRI-bias.xml'}. 
        @rtype: dict
        '''
        calIndex = {}
        
        for calFile in self.xmlIndex.keys():
            # calFile will look like 'GMOS-bias.xml', for split reference below.
            temp = calFile.split( "-" )
            adType = temp[0]
            tempCal = temp[1].split( '.' )[0]
            if tempCal == caltype:
                calIndex.update( {adType:calFile} )
        #print "CDL213:", calIndex
        return calIndex


    def __str__(self):
        return str(self.xmlIndex)