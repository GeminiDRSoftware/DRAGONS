from datetime import datetime
import pyfits as pf
from time import mktime
import urllib2 as ulib
#from mx.URL import *

from AstroData import AstroData
import ConfigSpace
import Descriptors
from ReductionObjectRequests import CalibrationRequest
#------------------------------------------------------------------------------ 

class CalibrationService( object ):
    '''
    Theoretically, if this is implemented, the search algorithms and retrieval will be here.
              
    '''
    
    calDirectory = None
    calList = []
    
    def __init__( self, calDirectoryURIs=["http://rallen:riverallen@chara/svn/DRSoftware/gemini_python/test_data/recipedata", 
                  "recipedata"], mode="local_disk" ):
        
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
        # 4)  when request received, basically thread pool or exec() a processRequest or One run CalService
        # 5)  Child sends out processedRequestMessage and is terminates
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
        Based on the info from the calibration request.
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
            #print "CS90: Checking if '" + calfile + "' is viable."
            ad = AstroData( calfile )
            desc = Descriptors.getCalculator( ad )
            
            if not self.searchIdentifiers( calRq.identifiers, desc, ad ):
                #print "FAILED IDENTIFIERS"
                continue
            if not self.searchCriteria( calRq.criteria, desc, ad ):
                #print "FAILED CRITERIA"
                continue
            #print "CS98: This '" + calfile + "' succeeded!"
            urilist.append( (calfile, desc, ad) )
            
        urilist = self.sortPriority( urilist, calRq.priorities )
            
        if urilist == []:
            # Nothing found, return None
            return None
        
        print "CS96 urilist --\n", urilist
        #"""
        return urilist
    
    
    def searchIdentifiers( self, identifiers, desc, ad ):
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
            if not self.compareProperty( {prop:identifiers[prop]}, desc, ad ):
                return False
        return True
    
    
    def searchCriteria( self, criteria, desc, ad, err=400000. ):
        '''
        Will perform the 'criteria' search  -- matching values must be identical or within tolerable error.
        
        @param identifiers: The identifier section from the request.
        @type identifiers: dict
        
        @param headers: List with all the headers for the fits file currently being searched.
        @type headers: list
        
        @param err: Theoretically, this might be used as some sort of below err threshhold in order for 
        it to be considered.
        @type err: float
        
        @return: True if all match, False otherwise.
        @rtype: boolean
        '''
        
        for prop in criteria.keys():
            compareRet = self.compareProperty( {prop:criteria[prop]}, desc, ad )
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
        
        @param listoffits: A list with a tuple containing the (calibration filename, Descriptor, Astrodata).
        This seems a bit 'perlish', but it makes the most sense at the time of creating this comment.
        @type listoffits: list
        
        @param priorities: Priorities from xml calibration file.
        @type priorities: dict
        
        @return: The sorted urllist (just the list of calibration URLs).
        @rtype: list
        '''
        
        # @@TODO: This entire sorting algorithm is incredibly inefficient, and needs to be changed
        # at some point.
        sortList = []
        for ffile, desc, ad in listoffits:
            sortvalue = []
            for prior in priorities.keys():
                compareRet = self.compareProperty( {prior:priorities[prior]}, desc, ad )
                sortvalue.append( compareRet )
            sortvalue.append( ffile )
            # Template: sortvalue - [prior1, prior2, prior3, ..., file]
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
    
    
    def compareProperty( self, prop, desc, ad):
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
        # Use the descriptor to obtain header key or 'tag'(i.e. filternames) values.
        retVal = desc.fetchValue( compareOn, ad )

        if compareOn.upper() == "OBSEPOCH":
            conRetVal = self.convertGemToUnixTime( retVal )
            conValue = self.convertGemToUnixTime( compValue )
            # Will return a single unix time value
            return abs( conRetVal - conValue )
        
        if type(compValue) == list:
            #print "LSC227 (list):", retVal, compValue 
            return compValue == retVal
        
        if typ.upper() == 'STRING':
            #print "LSC236 (str):", retVal.upper(), compValue.upper()
            return retVal.upper() == compValue.upper()

        elif typ.upper() == 'REAL':
            #print "LSC236 (real):", retVal, compValue
            return abs( float(retVal) - float(compValue) )
        else:
            raise "", typ, retVal
            return False
    
    
    def getCompareInfo( self, prop ):
        '''
        Unpacks the calibration request dict and tuple.
        '''
        compareOn = str( prop.keys()[0] )
        compareAttrs = prop.values()[0]
        extension = str( compareAttrs[0] )
        typ = str( compareAttrs[1] )
        value = compareAttrs[2]
        return compareOn, compareAttrs, extension, typ, value
    
    
    def convertGemToUnixTime( self, date, format="%Y-%m-%dT%H:%M:%S" ):
        '''
        This should convert a gemini time (in fits header) to a unix float time. 
        '''
        tempDate = date.split(".")[0]
        #print 'DATE:', tempDate
        t = datetime.strptime( tempDate, format )
        return mktime( t.timetuple() )
        
        
        
        
        
        
        