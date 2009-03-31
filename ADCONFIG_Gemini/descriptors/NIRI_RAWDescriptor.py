import Descriptors
import re
import GemCalcUtil
import Lookups

from Calculator import Calculator

stdkeyDictNIRI = {
    "key_niri_filter1":"FILTER1",
    "key_niri_filter2":"FILTER2",
    "key_niri_filter3":"FILTER3",
    }

class NIRI_RAWDescriptorCalc(Calculator):

    niriSpecDict = None
    
    def __init__(self):
        self.niriSpecDict = Lookups.getLookupTable("Gemini/NIRI/NIRISpecDict", "niriSpecDict")
        
    def filtername(self, dataset):
        """Return the Filtername for NIRI_RAW data.
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string, None if not present (implies corrupt data header)
        @return: the filtername
        """
        try:
            hdul = dataset.getHDUList()
            filt1  = hdul[0].header[stdkeyDictNIRI["key_niri_filter1"]]
            filt2  = hdul[0].header[stdkeyDictNIRI["key_niri_filter2"]]
            filt3  = hdul[0].header[stdkeyDictNIRI["key_niri_filter3"]]

            filt1 = GemCalcUtil.removeComponentID(filt1)
            filt2 = GemCalcUtil.removeComponentID(filt2)
            filt3 = GemCalcUtil.removeComponentID(filt3)

            # create list of filter vals
            filts = [filt1,filt2,filt3]
            #                           print "filts-0 = %s" % str(filts)
            # reject "open" "grism" and "pupil"
            #print filts
            filts2= []
            for f in filts:
                # print f
                if ("open" in f) or ("grism" in f) or ("pupil" in f):
                    pass
                else:
                    filts2.append(f)
            
            filts = filts2
                    
            #                           print "filts-1 = %s" % str(filts)
            
            # blank means an opaq mask was in place, which of course
            # blocks any other in place filters
            if "blank" in filts:
                retfilt = "blank"
                
            if len(filts) == 0:
                retfilt = "open"
            else:
                filts.sort()
                retfilt = ""
                first = True
                retfilt = "&".join(filts)
                
            # print "filter string NIRI: %s" % filtstr
        except KeyError:
            return None
        
        return retfilt
        
    def gain(self, dataset):
        return self.niriSpecDict["gain"]
        

