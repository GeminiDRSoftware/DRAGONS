import pyfits as pf

from ReductionObjectRequests import CalibrationRequest
import Descriptors
import AstroData
import ConfigSpace
from datetime import datetime
from time import mktime
import urllib2 as ulib
#from mx.URL import *

class CalibrationService( object ):
    '''
    Theoretically, if this is implemented, the search algorithms and retrieval will be here.
              
    '''
    
    calDirectory = None
    calList = []
    
    def __init__( self, calDirectoryURIs=["http://rallen:riverallen@chara/svn/DRSoftware/gemini_python/test_data/recipedata", 
                  "recipedata/calibrations"], mode="local_disk" ):
        
        # calList will contain absolute paths/filenames
        self.calList = []
        for path in calDirectoryURIs:
            self.calList.extend( ConfigSpace.generalWalk(path) )
        
        print "CS24:", self.calList
    
    
    def run(self):
        '''
        
        
        '''
        pass
        # Psuedo code
        # 1) set up, and send out appropriate messages
        # 2) while !not shutdown or restart message:
        # 3)  wait on message
        # 4)  when request received, basically thread pool or exec a processRequest
        # 5)  send out processedRequestMessage
        # 6) shutdown or restart
    
    def processRequest( self, message ):
        '''
        
        
        '''
        pass
        # Psuedo-code 
        # 1) Read Message
        # 2) run 'search'
        # 3) return results in output message for message bus.
    
    def search( self, calRq ):
        '''
        Based on the info from the calibration req
        '''
        
        inputfile = calRq.filename
        urilist = []
        
        
        """
        # What it looks like: Identifiers: {u'OBSTYPE': (u'PHU', u'string', u'BIAS')}
        caltype = calRq.identifiers['OBSTYPE'][2]
        
        if "N2009" in inputfile:
            if caltype == 'BIAS':
                urilist = ["./recipedata/N20090822S0207_bias.fits"]
            elif caltype == 'FLAT':
                urilist = ["./recipedata/N20090823S0102_flat.fits"]
        elif "N2002" in inputfile:
            if caltype == 'BIAS':
                urilist = ["./recipedata/N20020507S0045_bias.fits"]
            elif caltype == 'FLAT':
                urilist = ["./recipedata/N20020606S0149_flat.fits"]
        #"""
        
        
        #"""
        for calfile in self.calList:
            print "CS90: Checking if '" + calfile + "' is viable."
            headers = self.getAllHeaders( calfile )
            ad = AstroData( calfile )
            desc = Descriptors.getCalculator( ad )
            
            if not self.searchIdentifiers( calRq.identifiers, headers, desc, ad ):
                #print "FAILED IDENTIFIERS"
                continue
            if not self.searchCriteria( calRq.criteria, headers, desc, ad ):
                #print "FAILED CRITERIA"
                continue
            print "CS98: This '" + calfile + "' succeeded!"
            urilist.append( (calfile, headers, desc, ad) )
            
        urilist = self.sortPriority( urilist, calRq.priorities )
            
        if urilist == []:
            # Nothing found, return None
            return None
        
        print "CS96 urilist --\n", urilist
        #"""
        return urilist

    def searchIdentifiers( self, identifiers, headers, desc, ad ):
        '''
        Will perform the 'identifier' search  -- matching values must be identical.
        
        @param identifiers: The identifier section from the request.
        @type identifiers: dict
        
        @param headers: List with all the headers for the fits file currently being searched.
        @type headers: list
        
        @return: True if all match, False otherwise.
        @rtype: boolean
        '''
        
        for prop in identifiers.keys():
            if not self.compareProperty( {prop:identifiers[prop]}, headers ):
                return False
        return True
    
    
    def searchCriteria( self, criteria, headers, desc, ad, err=400000. ):
        '''
        Will perform the 'criteria' search  -- matching values must be identical or within tolerable error.
        
        @param identifiers: The identifier section from the request.
        @type identifiers: dict
        
        @param headers: List with all the headers for the fits file currently being searched.
        @type headers: list
        
        @return: True if all match, False otherwise.
        @rtype: boolean
        '''
        
        for prop in criteria.keys():
            compareRet = self.compareProperty( {prop:criteria[prop]}, headers )
            if type( compareRet ) == bool:
                if not compareRet:
                    #print "FAILED ON:", compareRet
                    return False
            else:
                if abs( compareRet ) > err:
                    #print "FAILED ON:", compareRet
                    return False
        return True
    
    def sortPriority( self, listoffits, priorities ):
        '''
        Will sort the listoffits based on the priorities.       .
        
        @param listoffits: A list with a tuple containing the (calibration filename, calibrations headers)
        @type listoffits: list
        
        @param priorities: Priorities from xml calibration file.
        @type priorities: dict
        
        @return: The sorted urllist
        @rtype: list
        '''
        
        # @@TODO: This entire sorting algorithm is incredibly inefficient, and needs to be changed
        # at some point.
        sortList = []
        for ffile, headers in listoffits:
            sortvalue = []
            for prior in priorities.keys():
                compareRet = self.compareProperty( {prior:priorities[prior]}, headers )
                sortvalue.append( compareRet )
            sortvalue.append( ffile )
            sortList.append( sortvalue )
        
        sortList.sort()
        ind = 0
        while ind < len( sortList ):
            # The list at this point would look something like
            # [sort property1, sort property2, ..., filename].
            # Thus, this just grabs the last filename.
            sortList[ind] = sortList[ind][-1]
            ind += 1
        
        print "CS172:", sortList
        return sortList

    def getAllHeaders( self, fname ):
        '''
        Returns a list with all the headers for a given fits file.
        
        @param fname: path/filename for the fits file
        @type fname: string
        
        @return: List with all the headers.
        @rtype: list
        '''
        try:
            fitsfile = pf.open( fname )
        except:
            raise "Could not open '" + str(fname) + "'."
        
        headerlist = []
        for ext in fitsfile:
            headerlist.append( ext.header )
            
        fitsfile.close()
        return headerlist

    def compareProperty( self, prop, headers, desc, ad):
        '''
        Compares a property (in xml calibration sense), to the headers of a calibration fits file.
        
        @param property: An xml calibration property of the form {Compare: (HeaderExt, type, Value)}
        @type property: dict
        
        @param headers: The list of headers. This should be [PHU, EXT1, EXT2, EXT3, ...etc]
        @type headers: list
        
        @return: The difference of the values if non-string, or True or False if string.
        @type: int, float, or None
        '''
        compareOn, compareAttrs, extension, typ, compValue = self.getCompareInfo( prop )
        ziggyTemp = desc.fetchValue( ad )
        print "ZIGGY:", ziggyTemp
        
        if compareOn.upper() == "OBSEPOCH":
            retVal = headers[extension]['DATE-OBS']
            retVal = retVal + "T" + headers[extension]['TIME-OBS']
            conRetVal = self.convertGemToUnixTime(retVal)
            conValue = self.convertGemToUnixTime(compValue)
            #print 'retval:', conRetVal
            #print 'comval:', conValue
            return abs( conRetVal - conValue )
        
        retVal = headers[extension][compareOn]
        print "NOSTRO:", retVal
        ziggyTemp = desc.fetchValue( ad )
        print "ZIGGY:", ziggyTemp
        
        if typ.upper() == 'STRING':
            #print "COMPARING: '"+retVal.upper()+"' '"+compValue.upper()+"'"
            if retVal.upper() == compValue.upper():
                return True
            
            return False 
        elif typ.upper() == 'REAL':
            #print "NUMS:", float(retVal), float(compValue)
            return abs( float(retVal) - float(compValue) )
        else:
            #print "SOMETHING BAD HERE:", typ, retVal
            return False
    
    def getCompareInfo( self, prop ):
        '''
        Descriptor stuff should go here.
        (ie) Custom Headers not actually in fits file (ie) FILTER, TIME, RDNOISE, ..etc
        
        This should theoretically return lists of stuff
        '''
        compareOn = str( prop.keys()[0] )
        compareAttrs = prop.values()[0]
        extension = self.getExtensionNumber( str(compareAttrs[0]) )
        typ = str( compareAttrs[1] )
        value = str( compareAttrs[2] )
        return compareOn, compareAttrs, extension, typ, value
        
        
    def getExtensionNumber( self, extensionName ):
        '''
        Takes the extension name and returns the corresponding extension number.
        
        @param extensionName: Extension value found in a xml calibration property. (i.e. 'PHU' or '[SCI,1]')
        @type extensionName: str
        
        @return: An integer value  
        @rtype: 
        '''
        
        if extensionName.upper() == 'PHU':
            return 0
        elif 'SCI,' in extensionName.upper():
            # 'SCI,1'
            return int( extensionName.split(',')[1] )  
        else:
            raise "Invalid extension passed: '" + extensionName + "'"
        
    def convertGemToUnixTime( self, date, format="%Y-%m-%dT%H:%M:%S" ):
        '''
        This should convert a gemini time (in fits header) to a unix float time. 
        '''
        tempDate = date.split(".")[0]
        #print 'DATE:', tempDate
        t = datetime.strptime( tempDate, format )
        return mktime( t.timetuple() )
        
        
        
        
        
        
        