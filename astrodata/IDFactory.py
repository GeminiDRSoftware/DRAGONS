import hashlib
import pyfits as pf
version_index = {"stackID":"1_0", "recipeID":"1_0", "displayID":"1_0"}


def generateStackableID( inputf, version ):
    '''
    
    
    '''
    if version != version_index['stackID']:
        try:
            idFunc = getattr( globals()['IDFactory'], 'generateStackID_' + version )
        except:
            raise "Version: '" + version + "' is either invalid or not supported." 
        
        return idFunc( inputf, version )
    
    shaObj = hashlib.sha1()
    phu = pf.getheader( inputf[0], 0 )
    shaObj.update( phu['OBSID'] )
    
    return shaObj.hexdigest()

def generateRecipeID( inputs, version ):
    '''
    
    
    '''
    if version != version_index['recipeID']:
        try:
            idFunc = getattr( globals()['IDFactory'], 'generateRecipeID_' + version )
        except:
            raise "Version: '" + version + "' is either invalid or not supported."
        
        return idFunc( inputs, version )
    
    shaObj = hashlib.sha1()
    for input in inputs: 
        shaObj.update( input )
        
    return shaObj.hexdigest()
    
def generateDisplayableID( inputf, version ):
    '''
    
    
    '''
    if version != version_index['displayID']:
        try:
            idFunc = getattr( globals()['IDFactory'], 'generateDisplayID_' + version )
        except:
            raise "Version: '" + version + "' is either invalid or not supported."
        
        return idFunc( inputf, version )
    
    shaObj = hashlib.sha1()
    phu = pf.getheader( inputf[0], 0 )
    shaObj.update( phu['OBSID'] )
    
    return shaObj.hexdigest()

    