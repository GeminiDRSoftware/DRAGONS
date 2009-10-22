import hashlib
import pyfits as pf
version_index = {"stackID":"1_0", "recipeID":"1_0", "displayID":"1_0"}


def generateStackableID( inputf, version = "1_0" ):
    '''
    Generate an ID from which all similar stackable data will have in common.
    
    @param inputf: Input Astrodatas 
    @type inputf: list of Astrodata instances.
    
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
    
    phu = pf.getheader( inputf[0], 0)
    ID = version + "_" + phu['OBSID'] + "_" + phu["OBJECT"]
    return ID
    # return shaObj.hexdigest()

def generateRecipeID( recname ):
    '''
    
    
    '''
    shaObj = hashlib.sha1()
    shaObj.update( recname )
        
    return shaObj.hexdigest()[:20]
    
def generateDisplayID( inputf, version ):
    '''
    
    
    '''
    if version != version_index['displayID']:
        try:
            # designed to call generateStackableID_
            idFunc = getattr( globals()['IDFactory'], 'generateDisplayID_' + version )
        except:
            raise "Version: '" + version + "' is either invalid or not supported." 
        
        return idFunc( inputf, version )
    
    if (False):
        shaObj = hashlib.sha1()
        phu = pf.getheader( inputf[0], 0 )
        shaObj.update( phu['OBSID'] )
        shaObj.update( phu['OBJECT'] )
    
    phu = pf.getheader( inputf[0], 0)
    ID = version + "_" + phu['OBSID'] + "_" + phu["OBJECT"]
    return ID
    # return shaObj.hexdigest()

    
