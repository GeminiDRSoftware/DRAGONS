from AstroData import AstroData
#from AstroDataType import *
from CalibrationRequestEvent import CalibrationRequestEvent
from xml.dom.minidom import parse
#from datetime import datetime
import os
import ConfigSpace
import pyfits

        
class CalibrationDefinitionLibrary( object ):
    '''
    This class deals with obtaining request data from XML calibration files.    
    '''    
    
    def __init__( self ):
        '''
        Goes into ConfigSpace and gets all the file URIs and creates an XML index.
        '''
        self.xmlIndex = {}
        try:
            for dpath, dnames, files in ConfigSpace.configWalk( "xmlcalibrations" ):
                for file in files:
                    self.xmlIndex.update( {file:os.path.join(str(dpath), file)} )
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
            ad = AstroData( input )       
            # print 'new image type', ad.getTypes( prune=False )
            leafTypes = ad.getTypes( prune=True )
            # print 'leaf types:', leafTypes
            typesWithCal = []
            for adType in leafTypes:
                filename = adType + "-" + caltype + ".xml"
                if filename in self.xmlIndex.keys():
                    typesWithCal.append( filename )
                else:
                    #search parents if no xml exist
                    try:
                        cl = ad.getClassificationLibrary()
                        #adType becomes dataClassification object
                        adType = cl.getTypeObj( adType )
                        parents = adType.getSuperTypes()
                        for supType in parents:
                            filename = supType.name + "-" + caltype + ".xml"
                            if filename in self.xmlIndex.keys():
                                typesWithCal.append( filename )
                                break                        
                    except:
                        print "Error getting adtype parents."
                        pass
                    
            print "TYPES W/ CALS:", typesWithCal
            typesWithCal = list( set( typesWithCal ) )
            if len( typesWithCal ) == 0:
                print "No Available calibration xml files for types", leafTypes
            elif len( typesWithCal ) > 1:
                print "More than one calibration xml file"
                print "CDL85:", str(typesWithCal) 
            
            filename = typesWithCal[0]
            try:
                calXMLURI = self.xmlIndex[filename]
                calXMLFile = open( calXMLURI, 'r' )
                xmlDom = parse( calXMLFile )
            except:
                raise "Error opening '%s'" %calXMLURI
            finally:
                calXMLFile.close() 
            #childNodes is the query tag           
            calReqEvent = self.parseQuery( xmlDom.childNodes[0], caltype, input )            
            reqEvents.append(calReqEvent)
        #goes to reduction context object to add to queue
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
       
        calReqEvent = CalibrationRequestEvent()
        calReqEvent.caltype = caltype
        query = xmlDomQueryNode
        if( query.hasAttribute("id") ):
            if( query.getAttribute("id") == caltype ):
                identifiers = query.getElementsByTagName( "identifier" )
                if len( identifiers ) > 0:
                    identifiers = identifiers[0]
                else:
                    raise "XML calibration has no identifiers"
                for child in identifiers.getElementsByTagName( "property" ):
                    key = child.getAttribute( "key" )                    
                    extension = child.getAttribute( "extension" )                    
                    elemType = child.getAttribute( "type" )                    
                    value = child.getAttribute( "value" ) 
                    
                    #creates dictionary object with multiple values               
                    calReqEvent.identifiers.update( {key:(extension,elemType,value)} ) 
                
                #creates hdulist for looking up header values                   
                inputHdulist = pyfits.open( input )                
                criteria = query.getElementsByTagName( "criteria" )
                if len( criteria ) > 0:
                    criteria = criteria[0]
                else:
                    raise "XML calibration has no identifiers"   
                for child in criteria.getElementsByTagName( "property" ):
                    key = child.getAttribute( "key" )                    
                    extension = child.getAttribute( "extension" )                    
                    elemType = child.getAttribute( "type" )
                    if extension == "PHU":
                        header = 0
                    else:
                        #split used to obtain science extension number from string, ie.  [SCI, 1]
                        header = int( extension.split(']')[0][-1] )                     
                    value = inputHdulist[header].header[str(key)]                                    
                    calReqEvent.criteria.update( {key:(extension,elemType,value)} )
                    
                priorities = query.getElementsByTagName( "priorities" )
                if len( priorities ) > 0:
                    priorities = priorities[0]
                else:
                    raise "XML calibration has no identifiers"    
                for child in priorities.getElementsByTagName( "property" ):
                    key = child.getAttribute( "key" )                    
                    extension = child.getAttribute( "extension" )                    
                    elemType = child.getAttribute( "type" )
                    #unfinished priority value                    
                    #value = child.getAttribute( "value" ) 
                    #this has to come from the input header                
                    calReqEvent.priorities.update( {key:(extension,elemType,value)} )            
        calReqEvent.filename = input                           
        return calReqEvent
                    
                
        

