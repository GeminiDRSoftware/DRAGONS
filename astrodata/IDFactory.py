import hashlib
import pyfits as pf
import re
#------------------------------------------------------------------------------ 
from astrodata.AstroData import AstroData
import Descriptors
#------------------------------------------------------------------------------ 
version_index = {"stackID":"1_0", "recipeID":"1_0", "display_id":"1_0"}

def generate_fingerprint( dataset, version = "1_0"):
    h = hashlib.md5()
    fullid = repr(dataset.types)+repr(dataset.all_descriptors())
    h.update(fullid)
    return h.hexdigest()
    
    
def generate_stackable_id( dataset, version = "1_0" ):
    '''
    Generate an ID from which all similar stackable data will have in common.
    
    @param dataset: Input AstroData instance or fits filename.
    @type dataset: AstroData instances or str
    
    @param version: The version from which to run.
    @type version: string
    
    @return: A stackable id.
    @rtype: string  
    '''
    if version != version_index['stackID']:
        try:
            # designed to call generateStackableID_
            idFunc = getattr( globals()['IDFactory'], 'generateStackableID_' + version )
        except:
            raise "Version: '" + version + "' is either invalid or not supported." 
        
        return idFunc( inputf, version )
    
    """
        shaObj = hashlib.sha1()
        phu = pf.getheader( inputf[0], 0 )
        shaObj.update( phu['OBSID'] )
        shaObj.update( phu['OBJECT'] )
    """
    
    if type(dataset) == str:
        phu = pf.getheader(dataset)
        ID = version + "_" + phu['OBSID'] + "_" + phu['OBJECT'] 
    elif type(dataset) == AstroData:
        ID = version + "_" + dataset.phuValue('OBSID') + "_" + dataset.phuValue('OBJECT')
        
    ID = make_id_safe_for_filename(ID)
    return ID
    # return shaObj.hexdigest()
  
def generate_display_id( dataset, version ):
    '''
    Generate an ID from which all similar stackable data will have in common.
    
    @param dataset: Input AstroData or fits filename
    @type dataset: list of AstroData instance
    
    @param version: The version from which to run.
    @type version: string
    
    @return: A display id.
    @rtype: string  
    '''
    if version != version_index['display_id']:
        try:
            # designed to call generateStackableID_
            idFunc = getattr( globals()['IDFactory'], 'generateDisplayID_' + version )
        except:
            raise "Version: '" + version + "' is either invalid or not supported." 
        
        return idFunc( inputf, version )
    
    """
        shaObj = hashlib.sha1()
        phu = pf.getheader( inputf[0], 0 )
        shaObj.update( phu['OBSID'] )
        shaObj.update( phu['OBJECT'] )
    """
    
    if type(dataset) == str:
        phu = pf.getheader(dataset)
        ID = version + "_" + phu['OBSID'] + "_" + phu['OBJECT'] 
    elif type(dataset) == AstroData:
        ID = version + "_" + dataset.phuValue('OBSID') + "_" + dataset.phuValue('OBJECT')
    return ID
    # return shaObj.hexdigest()


def generate_astro_data_id( dataset, version="1_0" ):
    '''
    An _id to be used to identify AstroData types. This is used for:
    
    1) Calibrations:
    
    Let's say a recipe performs
    
    getProcessedBias
    prepare
    biasCorrect
    
    Because of the prepare step, the calibration key determined at getProcessedBias will not 
    match biasCorrect because (N2009..., bias) will not match (gN2009..., bias). By using an astro_id,
    you can avoid this issue as you will have (DATALAB, bias). So, any steps inbetween getProcessedBias and
    biasCorrect will have no impact.
    
    2) Fringe:
    
    Fringe uses this as a FringeID, which is based off the first input of the list.
    
    @param dataset: Input AstroData instance or fits filename.
    @type dataset: AstroData instances or str
    
    @param version: The version from which to run.
    @type version: string
    
    @return: An astrodata id.
    @rtype: string  
    '''
    if type(dataset) == str:
        ad = AstroData( dataset )
        desc = Descriptors.get_calculator( ad )
        return desc.fetch_value('DATALAB', ad)
    elif type( dataset ) == AstroData:
        desc = Descriptors.get_calculator( dataset )
        return desc.fetch_value('DATALAB', dataset)
    else:
        raise "BAD ARGUMENT TYPE"
    
def generate_fringe_list_id( dataset, version='1_0' ):
    '''
    Generate an _id from which all similar stackable data will have in common.
    
    @param dataset: Input AstroData instance or fits filename.
    @type dataset: AstroData instances or str
    
    @param version: The version from which to run.
    @type version: string
    
    @return: A stackable id.
    @rtype: string
    '''
    return generate_stackable_id( dataset, version )
    
def make_id_safe_for_filename(_id):
    return re.sub(" ", "_", _id)

