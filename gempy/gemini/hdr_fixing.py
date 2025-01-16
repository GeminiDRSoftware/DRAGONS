# hdr_fixing.py
"""
Libraries of function to fix commissioning and early data headers.

Currently contains:
    - GMOS-N Hamamatsu commissioning header fixes.
       (command line script: gmosn_fix_headers)

"""


def gmosn_ham_fixes(hdulist, verbose):
    updated_glob = False

    updated = gmosn_ham_dateobs(hdulist, verbose)
    updated_glob |= updated

    return updated_glob

def gmosn_ham_dateobs(hdulist, verbose=False):
    """
    Correct DATE-OBS keyword in PHU.

    Parameters
    ----------
    hdulist : HDUList

    Returns
    -------
    A boolean indicating whether the header needed updating or not.

    """

    # DATE is normally okay.  Use that to correct DATE-OBS.
    date = hdulist[0].header['DATE']
    dateobs = hdulist[0].header['DATE-OBS']

    updated = False
    if date != dateobs:
        hdulist[0].header['DATE-OBS'] = date
        updated = True
        if verbose:
            print('DATE-OBS update the value of DATE, {}'.format(date))

    return updated
