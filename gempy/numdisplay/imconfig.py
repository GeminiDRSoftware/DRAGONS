"""
Version 1.0alpha - 9-Oct-2003 (WJH)

loadImtoolrc (imtoolrc=None):
    Locates, then reads in IMTOOLRC configuration file from
    system or user-specified location, and returns the
    dictionary for reference.

    The table gets loaded into a dictionary of the form:
        {configno:{'nframes':n,'width':nx,'height':ny},...}
    It can then be accessed using the syntax:
        fbtab[configno][attribute]
    For example:
        fbtab = loadImtoolrc()
        print fbtab[34]['width']
        1056 1024

"""

import os,string,sys

_default_imtoolrc_env = ["imtoolrc","IMTOOLRC"]
_default_system_imtoolrc = "/usr/local/lib/imtoolrc"
_default_local_imtoolrc = "imtoolrc"

def loadImtoolrc(imtoolrc=None):
    """
        Locates, then reads in IMTOOLRC configuration file from
        system or user-specified location, and returns the
        dictionary for reference.

    """
    # Find the IMTOOLRC file.  Except as noted below, this order
    # matches what ximtool and ds9 use.
    _home = os.getenv("HOME")

    # Look for path to directory where this module is installed
    # This will be last-resort location for IMTOOLRC that was
    # distributed with this module.
    _module_path = os.path.split(__file__)[0]

    ####
    # list of file names to look for; ok to have None to skip an entry
    _name_list = []

    # There are two environment variables that might set the location
    # of imtoolrc:

    # getenv('imtoolrc')
    _name_list.append(os.getenv(_default_imtoolrc_env[0]))

    # getenv('IMTOOLRC')
    _name_list.append(os.getenv(_default_imtoolrc_env[1]))

    # ~/.imtoolrc
    if 'HOME' in os.environ :
        _name_list.append( os.path.join(os.environ['HOME'], ".imtoolrc") )
    _name_list.append(sys.prefix+os.sep+_default_local_imtoolrc)

    # /usr/local/lib/imtoolrc
    _name_list.append(_default_system_imtoolrc)

    # $iraf/dev/imtoolrc - this is not in ds9 or NOAO's ximtool,
    # but it is in the AURA Unified Release ximtool.  This is the
    # one place on your system where you can be certain that
    # imtoolrc is really there.  Eventually, we will make a patch
    # to add this to ds9 and to IRAF.
    if 'iraf' in os.environ :
        _name_list.append( os.path.join( os.environ['iraf'], 'dev', 'imtoolrc') )

    # special to numdisplay: use imtoolrc that is in the package directory.
    # Basically, this is our way of having a built-in default table.
    _name_list.append(_module_path+os.sep+'imtoolrc')

    ####
    # Search all possible IMTOOLRC names in list
    # and open the first one found...
    for name in _name_list:
        try:
            if name:
                _fdin = open(name)
                break
        except OSError as error:
            pass

    #Parse the file, line by line and populate the dictionary
    _lines = _fdin.readlines()
    _fdin.close()

    # Build a dictionary for the entire IMTOOL table
    # It will be indexed by configno.
    fbdict = {}

    for line in _lines:
        # Strip out any blanks/tabs, Python 3 compat
        line = line.strip()
        # Ignore empty lines
        if len(line) > 1:
            _lsp = line.split()
            # Also, ignore comment lines starting with '#'
            if _lsp[0] != '#':
                configno = int(_lsp[0])
                _dict = {'nframes':int(_lsp[1]),'width':int(_lsp[2]),'height':int(_lsp[3]),'name':_lsp[5]}
                fbdict[configno] = _dict
    return fbdict

def help():
    print(__doc__)
