'''



'''


from astrodata.adutils import terminal
from astrodata.adutils.terminal import TerminalController, ProgressBar 
import time 

def importIt( imports=[], term=None, message='Loading...' ):
    '''
    When calling this, you need to update your locals() or globals()
    with the dict returned.
    
    If you modify this function, make sure to declare variables, etc before
    orig_keys.
    
    '''
    pb = None
    elapsed_time = 0
    st = 0
    et = 0
    
    if term is not None:
        try:
            pb = ProgressBar( term, message )
        except:
            pb = None
    
    if type( imports ) == str:
        imports = imports.split( '\n' )
    
    importsize = len( imports ) * 1.
    counter = 0
    imp = None
    orig_keys = None 
    orig_keys = locals().keys()
    for imp in imports:
        st = time.time()
        if pb is not None:
            pb.update( float(counter/importsize), imp )
            counter += 1
        exec( '%s' %(str(imp)) )
        et = time.time()
        elapsed_time += et - st
        
    if pb is not None:
        pb.update( 1., 'Done' )
    new_keys = list( set( locals().keys() ) - set( orig_keys ) )
    
    retkeys = {}
    for key in new_keys:
        retkeys.update( {key:locals()[key]} )
        
    return retkeys
