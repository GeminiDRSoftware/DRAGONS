from datetime import datetime

from astrodata import Descriptors
#------------------------------------------------------------------------------ 
class ReductionObjectRequest( object ):
    '''
    The parent of all Requests which contains members prevalent to all requests
    (i.e. timestamp).
    '''
    def __init__(self):
        self.ver = None
        self.timestamp = datetime.now()

    def __str__(self):
        '''
        
        '''
        tempStr = "\nVERSION: " + str( self.ver ) + \
                    "\nTIMESTAMP: " + str( self.timestamp ) + "\n"
                    
        return tempStr


class CalibrationRequest( ReductionObjectRequest ):
    '''
    The structure that stores the calibration parameters from the xml 
    calibration file.
    It is used by the control loop to be added to the request queue.
    '''
    filename = None
    identifiers = {}
    criteria = {}
    priorities = {}
    caltype = None
    
    def __init__( self,  filename=None, identifiers={}, criteria={}, priorities={}, caltype=None ):
        super( CalibrationRequest, self ).__init__()
        self.filename = None#filename
        self.identifiers = {}#identifiers
        self.criteria = {}#criteria
        self.priorities = {}#priorities
        self.caltype = None#caltype
        
    
    def __str__(self):
        tempStr = super( CalibrationRequest, self ).__str__()
        tempStr = tempStr + """filename: %(name)s
Identifiers: %(id)s
Criteria: %(crit)s
Priorities: %(pri)s
"""% {'name':str(self.filename),'id':str(self.identifiers), \
              'crit':str(self.criteria),'pri':str(self.priorities)}
        return tempStr

class DisplayRequest( ReductionObjectRequest ):
    '''
    The request to display a list of fits files.
    '''
    #disID - displayID - displayList
    def __init__( self, disID=None, disList=[] ):
        super( DisplayRequest, self ).__init__()
        self.disID = disID
        self.disList = disList
        
        
    def __str__(self):
        tempStr = super( DisplayRequest, self ).__str__()
        tempStr = tempStr + "ID: " + str( self.disID ) + "\n" + \
                    "DISPLAY LIST: " + str( self.disList )
        
        return tempStr

class GetStackableRequest( ReductionObjectRequest ):
    '''
    The request to get the stackable list. (More of a PRS issue as updating the stack 
    already does this.)
    '''
    def __init__( self, stkID=None ):
        super( GetStackableRequest, self ).__init__()
        self.stkID = stkID
        
    def __str__(self):
        tempStr = super( GetStackableRequest, self ).__str__()
        tempStr = tempStr + "ID: " + str( self.stkID )
        
        return tempStr

class ImageQualityRequest( ReductionObjectRequest ):
    '''
    A request to publish image quality metrics to the message bus or in the case
    of stand-alone mode, display overlays, etc. (Demo)
    '''
    def __init__( self, ad, ellMean, ellSigma, fWHMMean, fWHMSigma ):
        super( ImageQualityRequest, self ).__init__()
        #
        self.ad = ad
        self.filename = ad.filename
        self.ellMean = ellMean
        self.ellSigma = ellSigma
        self.fwhmMean = fWHMMean
        self.fwhmSigma = fWHMSigma
        desc = Descriptors.getCalculator( ad )
        self.pixelScale = desc.fetchValue( 'PIXSCALE', ad )
        self.seeing = self.fwhmMean# * self.pixelScale
        
    def __str__(self):
        tempStr = "-" * 40
        tempStr = tempStr + \
"""
Filename:           %(name)s
Ellipticity Mean:   %(emea)s 
Ellipticity Sigma:  %(esig)s                  
FWHM Mean:          %(fmea)s
FWHM Sigma:         %(fsig)s
Seeing:             %(seei)s
PixelScale:         %(pixs)s""" %{'name':self.filename, 'emea':self.ellMean, 'esig':self.ellSigma, 'fmea':self.fwhmMean,
      'fsig':self.fwhmSigma, 'seei':self.seeing, 'pixs':self.pixelScale}
        
        tempStr = tempStr + super( ImageQualityRequest, self ).__str__()
        
        return tempStr

class UpdateStackableRequest( ReductionObjectRequest ):
    '''
    Contains all relevant information to request updating the stackable index.
    '''
    def __init__( self, stkID=None, stkList=[] ):
        super( UpdateStackableRequest, self ).__init__()
        self.stkID = stkID
        self.stkList = stkList
    
        
    def __str__(self):
        tempStr = super( UpdateStackableRequest, self ).__str__()
        tempStr = tempStr + "ID: " + str( self.stkID ) + "\n" + \
                    "STACKABLE LIST: " + str( self.stkList )
        
        return tempStr

    
