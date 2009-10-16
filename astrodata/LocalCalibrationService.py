import pyfits as pf

from ReductionObjectRequests import CalibrationRequest
import ConfigSpace


class CalibrationService( object ):
    '''
    Theoretically, if this is implemented, the search algorithms and retrieval will be here.
    '''
    
    calDirectory = None
    calList = []
    
    WITHIN_ERROR = 10.
    
    def __init__( self, calDirectoryURIs=["recipedata/calibrations", "recipedata"], mode="local_disk" ):
        
        # This will be of the form:
        # {filename:path/filename}
        # This code has some definite bug possiblities, (ie if you have two filenames with the 
        # same name in different directories, then the last one to be processed is the only one
        # in the index.
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
        This is definitely going to change, and is entirely temporary.
        '''
        
        inputfile = calRq.filename
        urilist = []
        
        # What it looks like: Identifiers: {u'OBSTYPE': (u'PHU', u'string', u'BIAS')}
        #"""
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
        
        
        """
        for calfile in self.calList:
            headers = self.getAllHeaders( calfile )
            if not searchIdentifiers( calRq.identifiers ):
                continue
            if not searchCriteria( calRq.criteria ):
                continue
            urilist.append( (calfile, headers) )
        
        urilist = self.sortPriority( urilist, calRq.priorities )
            
        if urilist == []:
            # Nothing found, return None
            return None
        
        #"""
        return urilist

    def searchIdentifiers( self, identifiers, headers ):
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
            if not self.compareProperty( prop, headers ):
                return False
        return True
    
    
    def searchCriteria( self, criteria, headers ):
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
            compareRet = self.compareProperty( prop, headers )
            if type( compareRet ) == bool:
                if not compareRet:
                    return False
            else:
                if abs( compareRet ) > WITHIN_ERROR:
                    return False
        return True
    
    def sortPriority( self, listoffits, priorities ):
        '''
        Will sort the listoffits based on the priorities        .
        
        @param identifiers: The identifier section from the request.
        @type identifiers: dict
        
        @param headers: List with all the headers for the fits file currently being searched.
        @type headers: list
        
        @return: True if all match, False otherwise.
        @rtype: boolean
        '''
        
        # @@TODO: This entire sorting algorithm is incredibly inefficient, and needs to be changed
        # at some point.
        sortList = []
        for ffile, headers in listoffits:
            sortvalue = []
            for prior in priorities.keys():
                compareRet = self.compareProperty( prop, headers )
                sortvalue.append( compareRet )
            sortvalue.append( ffile )
            sortList.append( sortvalue )
        
        sortList.sort()
        ind = 0
        while ind < len( sortList ):
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

    def compareProperty( self, prop, headers):
        '''
        Compares a property (in xml calibration sense), to the headers of a calibration fits file.
        
        @param property: An xml calibration property of the form {Compare: (HeaderExt, type, Value)}
        @type property: dict
        
        @param headers: The list of headers. This should be [PHU, EXT1, EXT2, EXT3, ...etc]
        @type headers: list
        
        @return: The difference of the values if non-string, or True or False if string.
        @type: int, float, or None
        '''
        compareOn, compareAttrs, extension, typ, value = self.getCompareInfo( prop.keys()[0] )
        
        
    
    def getCompareInfo( self, prop ):
        '''
        Descriptor stuff should go here.
        (ie) Custom Headers not actually in fits file (ie) FILTER, TIME, RDNOISE, ..etc
        
        This should theoretically return lists of stuff
        '''
        compareOn = prop.keys()[0]
        compareAttrs = prop.values()
        extension = self.getExtensionNumber( compareAttrs[0] )
        typ = compareAttrs[1]
        value = compareAttrs[2]
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
        elif '[SCI,' in extensionName.upper():
            return int( extension.split(',')[0].split(']')[0])  
        else:
            raise "Invalid extension passed: '" + extensionName + "'"
        
        
    
        
        
        