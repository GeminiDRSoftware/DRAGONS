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

from astrodata import add_header_to_table
from astropy.table import Column, Table
from astroquery.vo_conesearch.conesearch import conesearch
from astroquery.vo_conesearch.exceptions import VOSError

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
    'sdss9_mko': "http://mkocatalog2/cgi-bin/conesearch.py?CATALOG=sdss9&",
    'sdss9_cpo': "http://cpocatalog2/cgi-bin/conesearch.py?CATALOG=sdss9&",
    'sdss9_vizier': "http://vizier.u-strasbg.fr/viz-bin/votable/-A?-source=sdss9&",
    '2mass_mko': "http://mkocatalog2/cgi-bin/conesearch.py?CATALOG=twomass_psc&",
    '2mass_cpo': "http://cpocatalog2/cgi-bin/conesearch.py?CATALOG=twomass_psc&",
    '2mass_vizier': "http://vizier.u-strasbg.fr/viz-bin/votable/-A?-source=B/2mass&",
    'ukidss9_mko': "http://mkocatalog2/cgi-bin/conesearch.py?CATALOG=ukidss&",
    'ukidss9_cpo': "http://cpocatalog2/cgi-bin/conesearch.py?CATALOG=ukidss&",
    'gmos_mko': "http://mkocatalog2/cgi-bin/conesearch.py?CATALOG=gmos&",
    'gmos_cpo': "http://cpocatalog2/cgi-bin/conesearch.py?CATALOG=gmos&",
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
# - eg different model fits magnitides - if we wish
# ***** These need to be in the same order as the list in our_cols *****
SERVER_COLMAP = {
    'sdss9_mko': ['objid', 'raj2000', 'dej2000', 'umag', 'umag_err',
                  'gmag', 'gmag_err', 'rmag', 'rmag_err', 'imag',
                  'imag_err', 'zmag', 'zmag_err'],
    'sdss9_cpo': ['objid', 'raj2000', 'dej2000', 'umag', 'umag_err',
                  'gmag', 'gmag_err', 'rmag', 'rmag_err', 'imag',
                  'imag_err', 'zmag', 'zmag_err'],
    'sdss9_vizier': ['objID', 'RAJ2000', 'DEJ2000', 'umag', 'e_umag',
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
        raise ValueError('Invalid Catalog')

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

    # astroquery 0.4 removed the pedantic keyword in favor of the config
    # item, and switched to returning an astropy table by default (hence
    # return_astropy_table=False below).
    # Another change is that conesearch returns None and issue of
    # NoResultsWarning if no results are found, instead of raising a
    # VOSError: https://github.com/astropy/astroquery/pull/1528
    try:
        try:
            table = conesearch((ra, dec), sr, verb=3, catalog_db=url,
                               return_astropy_table=False, verbose=False)
        except TypeError:
            # astroquery < 0.4
            table = conesearch((ra, dec), sr, verb=3, catalog_db=url,
                               pedantic=False, verbose=False)
    except VOSError:
        log.stdinfo("VO conesearch produced no results")
        return

    # Did we get any results?
    if table is None or table.is_empty() or len(table.array) == 0:
        log.stdinfo(f"No results returned from {server}")
        return

    # It turns out to be not viable to use UCDs to select the columns,
    # even for the id, ra, and dec. Even with vizier. <sigh>
    # The first column is our running integer column
    ret_table = Table([list(range(1, len(table.array[server_cols[0]])+1))],
                      names=('Id',), dtype=('i4',))

    ret_table.add_column(Column(table.array[server_cols[0]], name='Cat_Id',
                                dtype='a'))
    ret_table.add_column(Column(table.array[server_cols[1]], name='RAJ2000',
                                dtype='f8', unit='deg'))
    ret_table.add_column(Column(table.array[server_cols[2]], name='DEJ2000',
                                dtype='f8', unit='deg'))

    # Now the photometry columns
    for col in range(3, len(cols)):
        ret_table.add_column(Column(table.array[server_cols[col]], name=cols[col],
                                    dtype='f4', unit='mag', format='8.4f'))

    header = add_header_to_table(ret_table)
    header['CATALOG'] = (catalog.upper(), 'Origin of source catalog')
    # Add comments to the header to describe it
    header.add_comment('Source catalog derived from the {} catalog'
                       .format(catalog))
    header.add_comment('Source catalog fetched from server at {}'.format(url))
    header.add_comment('Delivered Table name from server:  {}'
                       .format(table.name))
    for col in range(len(cols)):
        header.add_comment('UCD for field {} is {}'.format(
            cols[col], table.get_field_by_id(server_cols[col]).ucd))
    return ret_table
