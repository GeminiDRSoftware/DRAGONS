import os

from datetime  import datetime

class ReductionObjectRequest(object):
    """ The parent of all Requests which contains members prevalent to all 
    requests (i.e. timestamp).
    """
    def __init__(self):
        self.ver = None
        self.timestamp = datetime.now()

    def __str__(self):
        tempStr = '\nVERSION: ' + str(self.ver) + \
                    '\nTIMESTAMP: ' + str(self.timestamp) + '\n'
        return tempStr

class CalibrationRequest(ReductionObjectRequest):
    """ The structure that stores the calibration parameters from the xml 
    calibration file. It is used by the control loop to be added to the 
    request queue.
    """
    filename = None
    identifiers = {}
    criteria = {}
    priorities = {}
    caltype = None
    datalabel = None
    source = None
    descriptors = None
    types = None
    calurl_dict = None
    
    def __init__(self,  filename=None, identifiers={}, criteria={}, 
                    priorities={}, caltype=None , source = 'all', ad = None):
                    
        super(CalibrationRequest, self).__init__()
        self.filename = None     #filename
        self.identifiers = {}    #identifiers
        self.criteria = {}       #criteria
        self.priorities = {}     #priorities
        self.caltype = None      #caltype
        self.source = source
        self.ad = ad
        
    def __str__(self):
        tempStr = super(CalibrationRequest, self).__str__()
        tempStr = tempStr + '''filename: %(name)s
Identifiers: %(id)s
   Criteria: %(crit)s
 Priorities: %(pri)s
Descriptors: %(des)s
      Types: %(types)s
'''% {'name':str(self.filename), 'id':str(self.identifiers), \
              'crit':str(self.criteria), 'pri':str(self.priorities),
              "des":repr(self.descriptors),
              "types":repr(self.types)}
        return tempStr

    def as_dict(self):
        retd = {}
        retd.update({'filename': self.filename,
                     'caltype': self.caltype,
                     'datalabel' : self.datalabel,
                     'source': self.source,
                     "descriptors": self.descriptors,
                     "types": self.types,
                     #"ad":self.ad
                     })
        if self.calurl_dict:
            retd["calurl_dict"] = self.calurl_dict
        return retd
        
    def from_dict(self, params):
        self.filename = params['filename'] if 'filename' in params else None
        self.caltype = params['caltype'] if 'caltype' in params else None
        self.datalabel = params['datalabel'] if 'datalabel' in params else None
        self.descriptors = params["descriptors"] if "descriptors" in params else None
        self.types = params["types"] if "types" in params else None
        self.ad = params["ad"] if "types" in params else None
        return


class DisplayRequest(ReductionObjectRequest):
    """ The request to display a list of fits files. """
    from astrodata import IDFactory

    def __init__(self, dis_id=None, dis_list=[]):
        super(DisplayRequest, self).__init__()
        self.disID = dis_id
        self.disList = dis_list
        
    def __str__(self):
        tempStr = super( DisplayRequest, self ).__str__()
        tempStr = tempStr + 'ID: ' + str(self.disID) + '\n' + \
                    'DISPLAY LIST: ' + str(self.disList)
        return tempStr
        
    def to_nested_dicts(self):
        d = {} # overall dict, single elements keyed with display
               # the command level
        f = {} # filename based dict
        tdis = {'files':f, 'type':'main'} # this display event level
        d.update({'display':tdis})
        for i in self.disList:
            filedict = {}
            f.update({os.path.basename(i.filename): filedict })
            filedict.update({'filename':i.filename})
            filedict.update({'display_id':IDFactory.generate_stackable_id(i.ad)})
        return d


class GetStackableRequest(ReductionObjectRequest):
    """ The request to get the stackable list. (More of a PRS issue as updating
    the stack already does this.)
    """
    def __init__(self, stk_id=None):
        super(GetStackableRequest, self).__init__()
        self.stk_id = stk_id
        
    def __str__(self):
        tempStr = super(GetStackableRequest, self).__str__()
        tempStr = tempStr + 'ID: ' + str(self.stk_id)
        return tempStr


class ImageQualityRequest(ReductionObjectRequest):
    """ A request to publish image quality metrics to the message bus 
    or in the case of stand-alone mode, display overlays, etc. (Demo)
    """
    def __init__( self, ad, ell_mean, ell_sigma, fwhm_mean, fwhm_sigma ):
        from astrodata import Descriptors
        
        super( ImageQualityRequest, self ).__init__()
        #
        self.ad = ad
        self.filename = ad.filename
        self.ellMean = ell_mean
        self.ellSigma = ell_sigma
        self.fwhmMean = fwhm_mean
        self.fwhmSigma = fwhm_sigma
        desc = Descriptors.get_calculator( ad )
        self.pixelScale = ad.pixel_scale().as_pytype()
        self.seeing = self.fwhmMean * self.pixelScale
        
    def __str__(self):
        tempStr = '-' * 40
        tempStr = tempStr + \
'''
Filename:           %(name)s
Ellipticity Mean:   %(emea)s 
Ellipticity Sigma:  %(esig)s                  
FWHM Mean:          %(fmea)s
FWHM Sigma:         %(fsig)s
Seeing:             %(seei)s
PixelScale:         %(pixs)s''' % {'name':self.filename, 
                                   'emea':self.ell_mean, 
                                   'esig':self.ell_sigma, 
                                   'fmea':self.fwhmMean,
                                   'fsig':self.fwhmSigma, 
                                   'seei':self.seeing, 
                                   'pixs':self.pixelScale}
        tempStr = tempStr + super(ImageQualityRequest, self).__str__()
        return tempStr


class UpdateStackableRequest(ReductionObjectRequest):
    """ Contains all relevant information to request updating the 
    stackable index.
    """
    def __init__(self, stk_id=None, stk_list=[]):
        super(UpdateStackableRequest, self).__init__()
        self.stk_id = stk_id
        self.stk_list = stk_list
    
    def __str__(self):
        tempStr = super(UpdateStackableRequest, self).__str__()
        tempStr = tempStr + 'ID: ' + str( self.stk_id ) + '\n' + \
                    'STACKABLE LIST: ' + str( self.stk_list )
        return tempStr
    
