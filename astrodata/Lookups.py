import os
import ConfigSpace
import pyfits

def get_lookup_table(modname, *lookup):
    """
        get_lookup_table() is used to get lookup table style sets of variables
        from a common facility, allowing the storage in common (global) space
        so that multiple scripts can refer to one lookup table
        without having to manage where this table is stored.  E.g. the Calculator
        (see L{Descriptors}) for NIRI data requires a NIRI lookup table that
        other parts of the package, unrelated to Descriptors, also need to 
        access.  This facility saves these separate components from knowing
        where the configuration is actually stored, or even that other
        parts of the system are relying on it, and ensure that changes will
        affect every part of the system.
        
    @param modname: namespace specifier for the table... in default case this
        is the directory and file name of the module in which the lookup
        table is stored, and the file is pure python.  However, the Lookups
        module can redirect this, using the modname, for example, as a
        key to find the lookup table in a database or elsewhere. Nothing like
        the latter is done at this time, and what is loaded are pure python
        files (e.g. a dict definition) from disk.
    @type modname: string
    @param lookup: name of the lookup table to load
    @type lookup: string
    """

    modname = ConfigSpace.lookup_path(modname)
    if ".py" in modname:
        f = file(modname)
      
        exec(f)
        f.close()

        if len(lookup) == 1:
            retval = eval (lookup[0])
        else:
            retval = []
            for item in lookup:
                retval.append(eval(item))
    elif ".fits" in modname:
        # in this case lookup will have extension ids
        table = pyfits.open(modname)
        if len(lookup) == 1:
            retval = table[lookup[0]]
        else:
            for item in lookup:
                retval.append(table[item])
            
    else:
        raise "this should never happen, tell someone"
    return retval
    
