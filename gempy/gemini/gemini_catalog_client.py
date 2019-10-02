#
#                                                                        DRAGONS
#
#                                                       gemini_catalog_client.py
# ------------------------------------------------------------------------------
from builtins import range

# ------------------------------------------------------------------------------
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
# ------------------------------------------------------------------------------
#
# Note: In astropy 3 (python 3 only support), the vo context functions moved
# from astropy.vo to astroquery. Under astroquery, former ImportError exceptions
# now throw a ModuleNotFoundError, an object not available in < astropy 3.0 and
# the older astropy.vo package.

# We handle potential import issues between astropy versions.

try:
    from astropy.vo.client.conesearch import conesearch as vo_conesearch
    from astropy.vo.client.vos_catalog import VOSError
except ImportError:
    from astroquery.vo_conesearch.conesearch import conesearch as vo_conesearch
    from astroquery.vo_conesearch.exceptions import VOSError

from astropy.table import Table, Column
from astropy.io import fits

from astrodata import add_header_to_table

from ..utils import logutils
# ------------------------------------------------------------------------------
# Used  to determine the function signature of the imported conesearch function
function_defaults = vo_conesearch.__defaults__
# ------------------------------------------------------------------------------
log = logutils.get_logger(__name__)
# ------------------------------------------------------------------------------
def get_fits_table(catalog, ra, dec, sr, server=None):
    """
    This function returns a QAP style REFCAT in the form of an astropy Table

    Parameters
    ----------
    catalog: str [sdss9 | 2mass | ukidss9 | gmos]
        name of catalog to search
    ra: float
        right ascension of search center, decimal degrees
    dec: float
        declination of search center, decimal degrees
    sr: float
        search radius, decimal degrees
    server: str/None
        name of server to query [sdss9_mko | sdss9_cpo | sdss9_vizier |
        2mass_mko | 2mass_cpo | 2mass_vizier | ukidss9_mko | ukidss9_cpo]
    Returns
    -------
    Table
        sources within the search cone
    """
    # This defines the list of available servers for each catalog. 
    cat_servers = {
        #'sdss9' : ['sdss9_mko', 'sdss9_cpo', 'sdss9_vizier'],
        #'2mass' : ['2mass_mko', '2mass_cpo', '2mass_vizier'],
        'sdss9' : ['sdss9_mko', 'sdss9_vizier'],
        '2mass' : ['2mass_mko', '2mass_vizier'],
        'ukidss9' : ['ukidss9_mko', 'ukidss9_cpo'],
        'gmos' : ['gmos_mko', 'gmos_cpo'],
    }

    # Check catalog given is valid
    assert catalog in cat_servers, 'Invalid Catalog'

    # Check server if given is valid
    assert server is None or server in cat_servers[catalog], 'Invalid Server'

    if server:
        cat_servers[catalog] = [server]

    fits_table = None
    for server in cat_servers[catalog]:
        fits_table = get_fits_table_from_server(catalog, server, ra, dec, sr)
        if fits_table:
            break
    return fits_table


def get_fits_table_from_server(catalog, server, ra, dec, sr):
    """
    This function fetches sources from the specified catalog from the specified
    server within the search radius and returns a Table containing them.

    Parameters
    ----------
    catalog: str [sdss9 | 2mass | ukidss9 | gmos]
        name of catalog to search
    server: str
        name of server to query [sdss9_mko | sdss9_cpo | sdss9_vizier |
        2mass_mko | 2mass_cpo | 2mass_vizier | ukidss9_mko | ukidss9_cpo]
    ra: float
        right ascension of search center, decimal degrees
    dec: float
        declination of search center, decimal degrees
    sr: float
        search radius, decimal degrees
    Returns
    -------
    Table
        sources within the search cone
    """
    # This defines the URL for each server
    # There must be an entry in this dictionary for each server
    # listed in cat_servers above
    server_urls = {
        'sdss9_mko': "http://mkocatalog2/cgi-bin/conesearch.py?CATALOG=sdss9&",
        'sdss9_cpo': "http://cpocatalog2/cgi-bin/conesearch.py?CATALOG=sdss9&",
        'sdss9_vizier': 
            "http://vizier.u-strasbg.fr/viz-bin/votable/-A?-source=sdss9&",
        '2mass_mko': 
            "http://mkocatalog2/cgi-bin/conesearch.py?CATALOG=twomass_psc&",
        '2mass_cpo': 
            "http://cpocatalog2/cgi-bin/conesearch.py?CATALOG=twomass_psc&",
        '2mass_vizier': 
            "http://vizier.u-strasbg.fr/viz-bin/votable/-A?-source=B/2mass&",
        'ukidss9_mko': 
            "http://mkocatalog2/cgi-bin/conesearch.py?CATALOG=ukidss&",
        'ukidss9_cpo': 
            "http://cpocatalog2/cgi-bin/conesearch.py?CATALOG=ukidss&",
        'gmos_mko': "http://mkocatalog2/cgi-bin/conesearch.py?CATALOG=gmos&",
        'gmos_cpo': "http://cpocatalog2/cgi-bin/conesearch.py?CATALOG=gmos&",
    }

    # This defines the column names *we* will use for that catalog.
    # There must be one entry in this list for each catalog listed
    # in cat_servers above
    cat_cols = {
        'sdss9' : ['catid', 'raj2000', 'dej2000', 'umag', 'umag_err', 
                   'gmag', 'gmag_err', 'rmag', 'rmag_err', 'imag', 
                   'imag_err', 'zmag', 'zmag_err'],
        '2mass' : ['catid', 'raj2000', 'dej2000', 'jmag', 'jmag_err', 
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
    server_colmap = {
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

    # OK, do the query
    url = server_urls[server]
    cols = cat_cols[catalog]
    server_cols = server_colmap[server]

    # print "RA, Dec, radius:", ra, dec, sr
    # print "catalog: %s" % catalog
    # print "server: %s" % server
    # print "url: %s" % url
    # print "cols       : %s" % cols
    # print "server_cols: %s" % server_cols
    # print "\n\n"

    # turn on verbose for debug to stdout. 
    # Need verb=3 to get the right cols from vizier

    # The following phrase is implemented to handle differing function 
    # signatures and return behaviours of vo conesearch function. Under 
    # astropy, conesearch throws a VOSError exception on no results. Which
    # seems a bit extreme. See the import phrase at top.
    try:
        table = vo_conesearch((ra,dec), sr, verb=3, catalog_db=url,
                              pedantic=False, verbose=False)
    except VOSError:
        log.stdinfo("VO conesearch produced no results")
        return None

    # Did we get any results?
    if(table.is_empty() or len(table.array) == 0):
        log.stdinfo("No results returned")
        return None

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
    header.add_comment('Source catalog derived from the {} catalog'.
                       format(catalog))
    header.add_comment('Source catalog fetched from server at {}'.format(url))
    header.add_comment('Delivered Table name from serer:  {}'.
                       format(table.name))
    for col in range(len(cols)):
        header.add_comment('UCD for field {} is {}'.format(cols[col],
                                   table.get_field_by_id(server_cols[col]).ucd))
    return ret_table
