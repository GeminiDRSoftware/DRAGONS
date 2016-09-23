from os import getcwd
from time import mktime
from datetime import datetime
#------------------------------------------------------------------------------ 
from astrodata import AstroData
from astrodata.utils import ConfigSpace
#------------------------------------------------------------------------------ 
def convertGemToUnixTime(date, form="%Y-%m-%dT%H:%M:%S"):
    """
    Convert a gemini time (in fits header) to a unix float time.
    """
    tempDate = date.split(".")[0]
    t = datetime.strptime(tempDate, form)
    return mktime(t.timetuple())

def get_compare_info(prop):
    """
    Unpacks the calibration request dict and tuple.
    """
    compareOn = str(prop.keys()[0])
    compareAttrs = prop.values()[0]
    extension = str(compareAttrs[0])
    typ = str(compareAttrs[1])
    value = compareAttrs[2]
    return compareOn, compareAttrs, extension, typ, value
    
#------------------------------------------------------------------------------ 
class CalibrationService(object):
    """
    If and/or when implemented, search algorithms and retrieval 
    will be here.
    """
    def __init__( self, cal_directory_uris=None, mode="local_disk" ):
        msg = "CalibrationService Not Available."
        raise NotImplementedError(msg)

        print ("Loading Calibration Service Available Directories")
        # calList will contain absolute paths/filenames
        self.calList = []
        if not cal_directory_uris:
            cal_directory_uris = getcwd()
        for path in cal_directory_uris:
            for cpath in ConfigSpace.general_walk(path, [".fits"]):
                self.calList.append(cpath)
    
    def run(self):
        """ 
        # Psuedo code
        # 1) set up, and send out appropriate messages
        # 2) while !not shutdown or restart message:
        # 3)   wait on message
        # 4)   when request received, basically thread pool or exec() 
        #      a process_request or One run CalService
        # 5) Child sends out processedRequestMessage and terminates
        # 6) shutdown or restart
        """
        pass
    
    def process_request(self, message):
        """
        # Psuedo-code 
        # 1) Read Message
        # 2) run 'search'
        # 3) return results in output message for message bus.        
        """
        pass

    def search(self, cal_rq):
        """
        Searches the various fits files collecting valid calibrations 
        and eventually returning a sorted list of based on the priorities.
        
        @param cal_rq: The Calibration Request. For this localized version,
        it contains only the most critical information, but on the PRS, 
        this would be a message.

        @type cal_rq: CalibrationRequest instance.
        @return:      A sorted list of calibration pathnames.
        @rtype:       <list>
        """
        from astrodata.interface import Descriptors
        urilist = []     
        
        for calfile in self.calList:
            ad = AstroData(calfile)
            desc = Descriptors.get_calculator(ad)
            if not self.search_identifiers(cal_rq.identifiers, desc, ad):
                continue
            if not self.search_criteria(cal_rq.criteria, desc, ad):
                continue
            urilist.append((calfile, desc, ad))
        urilist = self.sort_priority(urilist, cal_rq.priorities)
        
        if urilist == []:
            return None
        return urilist
    
    def search_identifiers(self, identifiers, desc, ad):
        """
        Will perform the 'identifier' search  -- matching values must be 
        identical.
        
        @param identifiers: The identifier section from the request.
        @type identifiers: dict
        
        @param headers: List with all the headers for the fits file currently 
                        being searched.
        @type headers: list
        
        @return: True if all match, False otherwise.
        @rtype: boolean
        """
        for prop in identifiers.keys():
            if not self.compare_property({prop:identifiers[prop]}, desc, ad):
                return False
        return True
    
    def search_criteria(self, criteria, desc, ad, err=400000.):
        """
        Will perform the 'criteria' search  -- matching values must be 
        identical or within tolerable error.
        
        @param identifiers: The identifier section from the request.
        @type identifiers: dict
        
        @param headers: List with all the headers for the fits file 
                        currently being searched.
        @type headers: list
        
        @param err: Theoretically, this might be used as some sort of below 
                    err threshhold in order for it to be considered.
        @type err: float
        
        @return: True if all match, False otherwise.
        @rtype:  <bool>
        """
        
        for prop in criteria.keys():
            compareRet = self.compare_property({prop:criteria[prop]}, desc, ad)
            if type( compareRet ) == bool:
                if not compareRet:
                    return False
            else:
                if abs( compareRet ) > err:
                    return False
        return True
    
    def sort_priority(self, listoffits, priorities):
        """
        Will sort the listoffits based on the priorities. 
        
        @param listoffits: A list with a tuple containing the 
                           (calibration filename, Descriptor, Astrodata).
                           This seems a bit 'perlish'...
        @type listoffits: list
        
        @param priorities: Priorities from xml calibration file.
        @type priorities: dict
        
        @return: The sorted urllist (just the list of calibration URLs).
        @rtype: list
        """
        sortList = []
        for ffile, desc, ad in listoffits:
            sortvalue = []
            for prior in priorities.keys():
                compareRet = self.compare_property({prior:priorities[prior]}, 
                                                   desc, ad)
                sortvalue.append(compareRet)
            sortvalue.append(ffile)
            sortList.append(sortvalue)
        sortList.sort() 
        ind = 0
        while ind < len(sortList):
            sortList[ind] = sortList[ind][-1]
            ind += 1
        return sortList
    
    def compare_property(self, prop, desc, ad):
        """
        Compares a property (in xml calibration sense), to the headers of a 
        calibration fits file.
        
        @param property: An xml calibration property of the form 
                        {Compare: (HeaderExt, type, Value)}
        @type property: dict
        
        @param headers: The list of headers. This should be 
                        [PHU, EXT1, EXT2, EXT3, ...etc]
        @type headers: list
        
        @return: The difference of the values if non-string, or 
                 True or False if string.
        @type: int, float, or None
        """
        compareOn, compareAttrs, extension, typ, compValue = get_compare_info(prop)
        retVal = desc.fetch_value(compareOn, ad)
        if compareOn.upper() == "OBSEPOCH":
            conRetVal = convertGemToUnixTime(retVal)
            conValue  = convertGemToUnixTime(compValue)
            # single unix time value
            return abs(conRetVal - conValue)
        
        if type(compValue) == list:
            if compareOn.upper() == "RONORIG":
                return compValue[0] == retVal [0]
            else:
                return compValue == retVal
        
        if typ.upper() == 'STRING':
            return str(retVal).upper() == str(compValue).upper()
        elif typ.upper() == 'REAL':
            return abs(float(retVal) - float(compValue))
        else:
            raise "", typ, retVal
        return
