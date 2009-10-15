from ReductionObjectRequests import CalibrationRequest

class CalibrationService( object ):
    '''
    Theoretically, if this is implemented, the search algorithms and retrieval will be here.
    '''
    def __init__( self ):
        pass
    
    
    def search( self, calRq ):
        '''
        This is definitely going to change, and is entirely temporary.
        '''
        
        inputfile = calRq.filename
        uri = None
        
        # What it looks like: Identifiers: {u'OBSTYPE': (u'PHU', u'string', u'BIAS')}
        caltype = calRq.identifiers['OBSTYPE'][2]
        
        if "N2009" in inputfile:
            if caltype == 'BIAS':
                uri = "./recipedata/N20090822S0207_bias.fits"
            elif caltype == 'FLAT':
                uri = "./recipedata/N20090823S0102_flat.fits"
        elif "N2002" in inputfile:
            if caltype == 'BIAS':
                uri = "./recipedata/N20020507S0045_bias.fits"
            elif caltype == 'FLAT':
                uri = "./recipedata/N20020606S0149_flat.fits"
        
        return [uri]
