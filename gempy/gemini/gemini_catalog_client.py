#
#                                                                       DRAGONS
#
#                                                      gemini_catalog_client.py
# -----------------------------------------------------------------------------

"""
This gemini_catalog_client module contains code to access various catalogs
on various catalog servers, providing a common interface to the caller,
despite the fact that different catalogs, and indeed the same catalog
on different servers may be in different formats, for example column
names for the same quantity may differ between servers

For a given catalog, it has the capability to fall-back through a
priority ordered list of servers if the primary server appears to be down.
- It will handle this even if the secondard servers have the catalog in
question in a different format to the primary.

"""
import warnings

import numpy as np

from astrodata import add_header_to_table
from astroquery.vo_conesearch.conesearch import conesearch
from astroquery.exceptions import NoResultsWarning

from ..utils import logutils

log = logutils.get_logger(__name__)

# List of available servers for each catalog.
CAT_SERVERS = {
    # 'sdss9' : ['sdss9_mko', 'sdss9_cpo', 'sdss9_vizier'],
    # '2mass' : ['2mass_mko', '2mass_cpo', '2mass_vizier'],
    'sdss9': ['sdss9_mko', 'sdss9_vizier'],
    '2mass': ['2mass_mko', '2mass_vizier'],
    'ukidss9': ['ukidss9_mko', 'ukidss9_cpo'],
    'gmos': ['gmos_mko', 'gmos_cpo'],
}

# This defines the URL for each server
# There must be an entry in this dictionary for each server
# listed in cat_servers above
SERVER_URLS = {
    'sdss9_mko': "http://gncatalog.gemini.edu/cgi-bin/conesearch.py?CATALOG=sdss9&",
    'sdss9_cpo': "http://gscatalog.gemini.edu/cgi-bin/conesearch.py?CATALOG=sdss9&",
    'sdss9_vizier': "http://vizier.u-strasbg.fr/viz-bin/votable/-A?-source=sdss9&",
    '2mass_mko': "http://gncatalog.gemini.edu/cgi-bin/conesearch.py?CATALOG=twomass_psc&",
    '2mass_cpo': "http://gscatalog.gemini.edu/cgi-bin/conesearch.py?CATALOG=twomass_psc&",
    '2mass_vizier': "http://vizier.u-strasbg.fr/viz-bin/votable/-A?-source=2mass&",
    'ukidss9_mko': "http://gncatalog.gemini.edu/cgi-bin/conesearch.py?CATALOG=ukidss&",
    'ukidss9_cpo': "http://gscatalog.gemini.edu/cgi-bin/conesearch.py?CATALOG=ukidss&",
    'gmos_mko': "http://gncatalog.gemini.edu/cgi-bin/conesearch.py?CATALOG=gmos&",
    'gmos_cpo': "http://gscatalog.gemini.edu/cgi-bin/conesearch.py?CATALOG=gmos&",
}

# This defines the column names *we* will use for that catalog.
# There must be one entry in this list for each catalog listed
# in cat_servers above
CAT_COLS = {
    'sdss9': ['catid', 'raj2000', 'dej2000', 'umag', 'umag_err',
              'gmag', 'gmag_err', 'rmag', 'rmag_err', 'imag',
              'imag_err', 'zmag', 'zmag_err'],
    '2mass': ['catid', 'raj2000', 'dej2000', 'jmag', 'jmag_err',
              'hmag', 'hmag_err', 'kmag', 'kmag_err'],
    'ukidss9': ['catid', 'raj2000', 'dej2000', 'ymag', 'ymag_err',
                'zmag', 'zmag_err', 'jmag', 'jmag_err',
                'hmag', 'hmag_err', 'kmag', 'kmag_err'],
    'gmos': ['name', 'raj2000', 'dej2000', 'umag', 'umag_err',
             'gmag', 'gmag_err', 'rmag', 'rmag_err', 'imag',
             'imag_err', 'zmag', 'zmag_err']
}

# This defines the column name mapping for each catalog server to our
# column names. This copes with both variable server conventions, and
# also allows us to point to different columns in the upstream catalog
# - eg different model fits magnitudes - if we wish
# ***** These need to be in the same order as the list in our_cols *****
SERVER_COLMAP = {
    'sdss9_mko': ['objid', 'raj2000', 'dej2000', 'umag', 'umag_err',
                  'gmag', 'gmag_err', 'rmag', 'rmag_err', 'imag',
                  'imag_err', 'zmag', 'zmag_err'],
    'sdss9_cpo': ['objid', 'raj2000', 'dej2000', 'umag', 'umag_err',
                  'gmag', 'gmag_err', 'rmag', 'rmag_err', 'imag',
                  'imag_err', 'zmag', 'zmag_err'],
    'sdss9_vizier': ['objID', '_RAJ2000', '_DEJ2000', 'umag', 'e_umag',
                     'gmag', 'e_gmag', 'rmag', 'e_rmag', 'imag', 'e_imag',
                     'zmag', 'e_zmag'],
    '2mass_mko': ['designation', 'ra', 'decl', 'j_m', 'j_cmsig',
                  'h_m', 'h_cmsig', 'k_m', 'k_cmsig'],
    '2mass_cpo': ['designation', 'ra', 'decl', 'j_m', 'j_cmsig',
                  'h_m', 'h_cmsig', 'k_m', 'k_cmsig'],
    '2mass_vizier': ['_2MASS', 'RAJ2000', 'DEJ2000', 'Jmag', 'Jcmsig',
                     'Hmag', 'Hcmsig', 'Kmag', 'Kcmsig'],
    'ukidss9_mko': ['id', 'raj2000', 'dej2000', 'y_mag', 'y_mag_err',
                    'z_mag', 'z_mag_err', 'j_mag', 'j_mag_err',
                    'h_mag', 'h_mag_err', 'k_mag', 'k_mag_err'],
    'ukidss9_cpo': ['id', 'raj2000', 'dej2000', 'y_mag', 'y_mag_err',
                    'z_mag', 'z_mag_err', 'j_mag', 'j_mag_err',
                    'h_mag', 'h_mag_err', 'k_mag', 'k_mag_err'],
    'gmos_mko': ['name', 'raj2000', 'dej2000', 'umag', 'umag_err',
                 'gmag', 'gmag_err', 'rmag', 'rmag_err', 'imag',
                 'imag_err', 'zmag', 'zmag_err'],
    'gmos_cpo': ['name', 'raj2000', 'dej2000', 'umag', 'umag_err',
                 'gmag', 'gmag_err', 'rmag', 'rmag_err', 'imag',
                 'imag_err', 'zmag', 'zmag_err']
}


def get_fits_table(catalog, ra, dec, sr, server=None, verbose=False):
    """
    This function returns a QAP style REFCAT in the form of an astropy Table.

    Parameters
    ----------
    catalog : {'sdss9', '2mass', 'ukidss9', 'gmos'}
        Name of catalog to search.
    ra : float
        Right ascension of search center, decimal degrees.
    dec : float
        Declination of search center, decimal degrees.
    sr : float
        Search radius, decimal degrees.
    server : str, optional
        Name of server to query. Must be a valid server name in the following
        list: sdss9_mko, sdss9_cpo, sdss9_vizier, 2mass_mko, 2mass_cpo,
        2mass_vizier, ukidss9_mko, ukidss9_cpo.
    verbose : bool
        Verbose output.

    Returns
    -------
    `astropy.table.Table`
        Sources within the search cone.

    """
    # Check catalog given is valid
    if catalog not in CAT_SERVERS:
        raise ValueError('Invalid Catalog. Valid catalogs are: '+
                         ' '.join(CAT_SERVERS))

    # Check server if given is valid
    if server is not None and server not in CAT_SERVERS[catalog]:
        raise ValueError('Invalid Server')

    servers = [server] if server else CAT_SERVERS[catalog]

    for server in servers:
        tbl = get_fits_table_from_server(catalog, server, ra, dec, sr,
                                         verbose=verbose)
        if tbl is not None:
            return tbl


def get_fits_table_from_server(catalog, server, ra, dec, sr, verbose=False):
    """
    This function fetches sources from the specified catalog from the specified
    server within the search radius and returns a Table containing them.

    Parameters
    ----------
    catalog : {'sdss9', '2mass', 'ukidss9', 'gmos'}
        Name of catalog to search.
    server : str
        Name of server to query [sdss9_mko | sdss9_cpo | sdss9_vizier |
        2mass_mko | 2mass_cpo | 2mass_vizier | ukidss9_mko | ukidss9_cpo]
    ra : float
        Right ascension of search center, decimal degrees.
    dec : float
        Declination of search center, decimal degrees.
    sr : float
        Search radius, decimal degrees.
    verbose : bool
        Verbose output.

    Returns
    -------
    `astropy.table.Table`
        Sources within the search cone.

    """
    # OK, do the query
    url = SERVER_URLS[server]
    cols = CAT_COLS[catalog]
    server_cols = SERVER_COLMAP[server]

    if verbose:
        print("RA, Dec, radius :", ra, dec, sr)
        print("catalog         :", catalog)
        print("server          :", server)
        print("url             :", url)
        print("cols            :", cols)
        print("server_cols     :", server_cols)
        print("\n\n")

    # turn on verbose for debug to stdout.
    # Need verb=3 to get the right cols from vizier

    # Another change is that conesearch returns None and issue of
    # NoResultsWarning if no results are found, instead of raising a
    # VOSError: https://github.com/astropy/astroquery/pull/1528
    with warnings.catch_warnings(record=True) as warning_list:
        table = conesearch((ra, dec), sr, verb=3, catalog_db=url,
                           return_astropy_table=True, verbose=False)

    # Did we get any results?
    if not table:
        try:
            warning = warning_list[0]
        except IndexError:
            log.stdinfo(f"No results returned from {server} but no warning "
                        "issued")
        else:
            if warning.category == NoResultsWarning:
                log.stdinfo(f"No results returned from {server}")
            elif ("retries" in str(warning.message) or
                  "timed out" in str(warning.message)):
                log.warning(f"Server {server} appears to be down")
            else:
                log.warning(f"Unexpected warning from {server}: "
                            f"{warning.message}")
        return

    if server == 'sdss9_vizier':
        # Vizier uses the photoObj table from SDSS9, whereas the internal
        # server uses the calibObj, AKA "datasweep", which contains a subset
        # of photoObj, "designed for those who want to work with essentially
        # every well measured object, but only need the most commonly used
        # parameters".
        #
        # To get results similar to calibObj, we filter below on mode=1 to get
        # only the primary sources (the 'main' photometric observation of an
        # object). calibObj also uses a cut on magnitudes (see
        # http://www.sdss3.org/dr9/imaging/catalogs.php) but this is difficult
        # to reproduce here since the cuts apply to extinction-corrected
        # magnitudes, and we don't have the extinction values in the Vizier
        # table.
        table = table[table['mode'] == 1]

    # It turns out to be not viable to use UCDs to select the columns,
    # even for the id, ra, and dec. Even with vizier. <sigh>
    # The first column is our running integer column
    table.keep_columns(server_cols)
    table.add_column(np.arange(1, len(table)+1, dtype=np.int32),
                     name='Id', index=0)
    for old_name, new_name in zip(
            server_cols, ["Cat_Id", "RAJ2000", "DEJ2000"] + cols[3:]):
        table.rename_column(old_name, new_name)

    # Reorder columns and set datatypes
    table = table[['Id', 'Cat_Id', 'RAJ2000', 'DEJ2000'] + cols[3:]]
    table['Cat_Id'] = [str(item) for item in table['Cat_Id']]  # force string
    for c in table.colnames[4:]:
        table[c].format = "8.4f"

    table_name = table.meta.get('name')
    table.meta = None
    header = add_header_to_table(table)
    header['CATALOG'] = (catalog.upper(), 'Origin of source catalog')

    # Add comments to the header to describe it
    header.add_comment(f'Source catalog derived from the {catalog} catalog')
    header.add_comment(f'Source catalog fetched from server at {url}')
    if table_name:
        header.add_comment(f'Delivered Table name from server:  {table_name}')

    return table
