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
    'sdss9': ['sdss9_mko', 'sdss9_vizier', 'sdss9_viziercfa'],
    '2mass': ['2mass_mko', '2mass_vizier', '2mass_viziercfa'],
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
    'sdss9_viziercfa': "http://vizier.cfa.harvard.edu/viz-bin/votable/-A?-source=sdss9&",
    '2mass_mko': "http://mkocatalog2/cgi-bin/conesearch.py?CATALOG=twomass_psc&",
    '2mass_cpo': "http://cpocatalog2/cgi-bin/conesearch.py?CATALOG=twomass_psc&",
    '2mass_vizier': "http://vizier.u-strasbg.fr/viz-bin/votable/-A?-source=2mass&",
    '2mass_viziercfa': "http://vizier.cfa.harvard.edu/viz-bin/votable/-A?-source=2mass&",
    'ukidss9_mko': "http://mkocatalog2/cgi-bin/conesearch.py?CATALOG=ukidss&",
    'ukidss9_cpo': "http://cpocatalog2/cgi-bin/conesearch.py?CATALOG=ukidss&",
    'gmos_mko': "http://mkocatalog2/cgi-bin/conesearch.py?CATALOG=gmos&",
    'gmos_cpo': "http://cpocatalog2/cgi-bin/conesearch.py?CATALOG=gmos&",
}

# This defines the column names *we* will use for that catalog.
# There must be one entry in this list for each catalog listed
# in cat_servers above
# TODO replace with metaclass list of bands
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
    'sdss9_viziercfa': ['objID', '_RAJ2000', '_DEJ2000', 'umag', 'e_umag',
                        'gmag', 'e_gmag', 'rmag', 'e_rmag', 'imag', 'e_imag',
                        'zmag', 'e_zmag'],
    '2mass_mko': ['designation', 'ra', 'decl', 'j_m', 'j_cmsig',
                  'h_m', 'h_cmsig', 'k_m', 'k_cmsig'],
    '2mass_cpo': ['designation', 'ra', 'decl', 'j_m', 'j_cmsig',
                  'h_m', 'h_cmsig', 'k_m', 'k_cmsig'],
    '2mass_vizier': ['_2MASS', 'RAJ2000', 'DEJ2000', 'Jmag', 'Jcmsig',
                     'Hmag', 'Hcmsig', 'Kmag', 'Kcmsig'],
    '2mass_viziercfa': ['_2MASS', 'RAJ2000', 'DEJ2000', 'Jmag', 'Jcmsig',
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


class CatServerMetaclass(type):
    """
    Metaclass for cleaner definition of catalog servers via CatServer subclasses.

    This metaclass primarily sidesteps large boilerplate cat_cols definitions by
    allowing them to be encoded as a list of letter bands.  These are then extracted
    into `xmag` and `xmag_err` as well as supplying the initial columns.  Classes
    can also customize the first column name with a `first_cat_col` field or just
    accept the 'catid' default.
    """
    def __new__(cls, clsname, bases, attrs):
        def gen_cat_cols(val):
            first_cat_col = attrs.get('first_cat_col', 'catid')
            cat_cols = [first_cat_col, 'raj2000', 'dej2000']
            for c in val:
                cat_cols.extend([f'{c}mag', f'{c}mag_err'])
            return cat_cols
        def check_modify(k, v):
            if k == 'cat_cols' and isinstance(v, str):
                return gen_cat_cols(v)
            return v
        modified_attrs = {k: check_modify(k, v) for k, v in attrs.items()}
        return super(CatServerMetaclass, cls).__new__(
            cls, clsname, bases, modified_attrs)


class CatServer(object, metaclass=CatServerMetaclass):
    """
    Base class for catalog servers.

    This is the base class for the various catalog servers.  Subclasses are
    defined per-catalog.  The subclass should define the local column ids via
    `cat_cols` which is generally just the list of band letters.  `default_server_cols`
    should hold the list of server-side column ids.

    Instances can override the server column ids via the constructor, as well as
    providing a URL specific to that server.
    """
    default_server_cols = []
    cat_cols = None
    first_cat_col = 'catid'

    def __init__(self, url, server_cols=None):
        """
        Create a catalog server instance for accessing the specified URL and returning
        the default column IDs for this type, or specify custom IDs if needed.

        Parameters
        ----------
        url : str
            URL of remote server to access
        server_cols : list
            list of column IDs expected from remote server
        """
        if server_cols is None:
            self.server_cols = self.__class__.default_server_cols
        else:
            self.server_cols = server_cols
        if len(self.server_cols) != len(self.__class__.cat_cols):
            raise ValueError("Server columns and catalog columns should have the same length.")
        self.url = url


class SDSS9CatServer(CatServer):
    cat_cols = 'ugriz'
    default_server_cols = ['objid', 'raj2000', 'dej2000', 'umag', 'umag_err',
                           'gmag', 'gmag_err', 'rmag', 'rmag_err', 'imag',
                           'imag_err', 'zmag', 'zmag_err']


class TwoMassCatServer(CatServer):
    cat_cols = 'jhk'
    default_server_cols = ['designation', 'ra', 'decl', 'j_m', 'j_cmsig',
                           'h_m', 'h_cmsig', 'k_m', 'k_cmsig']


class UKIDSS9CatServer(CatServer):
    cat_cols = 'yzjhk'
    default_server_cols = ['id', 'raj2000', 'dej2000', 'y_mag', 'y_mag_err',
                           'z_mag', 'z_mag_err', 'j_mag', 'j_mag_err',
                           'h_mag', 'h_mag_err', 'k_mag', 'k_mag_err']


class GMOSCatServer(CatServer):
    cat_cols = 'ugriz'
    first_cat_col = 'name'
    default_server_cols = ['name', 'raj2000', 'dej2000', 'umag', 'umag_err',
                           'gmag', 'gmag_err', 'rmag', 'rmag_err', 'imag',
                           'imag_err', 'zmag', 'zmag_err']


# These are handy column ID lists for customizing the Vizier services for their expected server-side columns
_sdss9_viziercols = ['objID', '_RAJ2000', '_DEJ2000', 'umag', 'e_umag',
                     'gmag', 'e_gmag', 'rmag', 'e_rmag', 'imag', 'e_imag',
                     'zmag', 'e_zmag']
_twomass_viziercols = ['_2MASS', 'RAJ2000', 'DEJ2000', 'Jmag', 'Jcmsig',
                       'Hmag', 'Hcmsig', 'Kmag', 'Kcmsig']


# Defined Catalog Servers
#
# This is the set of defined catalog services for querying.  It is arranged in a dictioanry
# by catalog name, followed by server name.
#
# To define a new server for an existing catalog, it is likely sufficient to just add a new
# entry here using that class.  You can supply the new URL and, if needed, you can provide
# an alternate list of column names expected from the remote service.
#
# If you have a new catalog entirely, you will want to subclass CatServer above.  Use the
# other classes as a guide, but essentially you will:
#  * subclass CatServer
#  * provide a list of bands in a string to be expanded into the catalog columns (or an explicit list of names works)
#  * provide default_server_cols for the list of column names typically returned by the remote services
#  * (optional) provide first_cat_col with the first catalog column name if necessary (mainly a logging issue)
_cat_servers = {
    'sdss9': {
        'mko': SDSS9CatServer(url='http://gncatalog.gemini.edu/cgi-bin/conesearch.py?CATALOG=sdss9&'),
        # 'cpo': SDSS9CatServer(url='http://gscatalog.gemini.edu/cgi-bin/conesearch.py?CATALOG=sdss9&'),
        'vizier': SDSS9CatServer(server_cols=_sdss9_viziercols,
                                 url='http://vizier.u-strasbg.fr/viz-bin/votable/-A?-source=sdss9&'),
        'viziercfa': SDSS9CatServer(server_cols=_sdss9_viziercols,
                                    url='http://vizier.cfa.harvard.edu/viz-bin/votable/-A?-source=sdss9&'),
    },
    '2mass': {
        'mko': TwoMassCatServer(url='http://gncatalog.gemini.edu/cgi-bin/conesearch.py?CATALOG=twomass_psc&'),
        # 'cpo': TwoMassCatServer(url='http://gscatalog.gemini.edu/cgi-bin/conesearch.py?CATALOG=twomass_psc&'),
        'vizier': TwoMassCatServer(server_cols=_twomass_viziercols,
                                   url='http://vizier.u-strasbg.fr/viz-bin/votable/-A?-source=2mass&'),
        'viziercfa': TwoMassCatServer(server_cols=_twomass_viziercols,
                                      url='http://vizier.cfa.harvard.edu/viz-bin/votable/-A?-source=2mass&'),
    },
    'ukidss9': {
        'mko': UKIDSS9CatServer(url='http://gncatalog.gemini.edu/cgi-bin/conesearch.py?CATALOG=ukidss&'),
        'cpo': UKIDSS9CatServer(url='http://gscatalog.gemini.edu/cgi-bin/conesearch.py?CATALOG=ukidss&'),
    },
    'gmos': {
        'mko': GMOSCatServer(url='http://gncatalog.gemini.edu/cgi-bin/conesearch.py?CATALOG=gmos&'),
        'cpo': GMOSCatServer(url='http://gscatalog.gemini.edu/cgi-bin/conesearch.py?CATALOG=gmos&'),
    }
}


def get_fits_table(catalog, ra, dec, sr, server=None, verbose=False, timeout=None):
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
    timeout : float
        Timeout for remote queries in seconds, None for default from `astroquery`

    Returns
    -------
    `astropy.table.Table`
        Sources within the search cone.

    """
    # Check catalog given is valid
    if catalog not in _cat_servers:
        raise ValueError('Invalid Catalog. Valid catalogs are: '+
                         ' '.join(_cat_servers))

    # Check server if given is valid
    check_server = server
    if check_server is not None and catalog is not None and check_server.startswith(catalog):
        check_server = check_server[len(catalog)+1:]
    if server is not None and check_server not in _cat_servers[catalog]:
        raise ValueError('Invalid Server')

    servers = [server] if server else _cat_servers[catalog].keys()

    for server in servers:
        tbl = get_fits_table_from_server(catalog, server, ra, dec, sr,
                                         verbose=verbose, timeout=timeout)
        if tbl is not None:
            return tbl


def get_fits_table_from_server(catalog, server, ra, dec, sr, verbose=False, timeout=None):
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
    timeout : float
        Timeout for remote services in seconds, None for default from `astroquery`

    Returns
    -------
    `astropy.table.Table`
        Sources within the search cone.

    """
    # OK, do the query
    svr = _cat_servers[catalog][server]
    # url = svr.url
    # cols = svr.cat_cols
    # server_cols = svr.server_cols

    if verbose:
        print("RA, Dec, radius :", ra, dec, sr)
        print("catalog         :", catalog)
        print("server          :", server)
        print("url             :", svr.url)
        print("cols            :", svr.cat_cols)
        print("server_cols     :", svr.server_cols)
        print("\n\n")

    # turn on verbose for debug to stdout.
    # Need verb=3 to get the right cols from vizier

    # astroquery 0.4 removed the pedantic keyword in favor of the config
    # item, and switched to returning an astropy table by default (hence
    # return_astropy_table=False below).
    # Another change is that conesearch returns None and issue of
    # NoResultsWarning if no results are found, instead of raising a
    # VOSError: https://github.com/astropy/astroquery/pull/1528
    from astroquery.vo_conesearch import conf
    if timeout is None:
        timeout = conf.timeout  # just use current value, keeps the logic sane
    with conf.set_temp("timeout", timeout):  # It's a kind of magic
        try:
            try:
                table = conesearch((ra, dec), sr, verb=3, catalog_db=svr.url,
                                   return_astropy_table=False, verbose=False)
            except TypeError:
                # astroquery < 0.4
                table = conesearch((ra, dec), sr, verb=3, catalog_db=svr.url,
                                   pedantic=False, verbose=False)
        except VOSError:
            log.stdinfo("VO conesearch produced no results")
            return

    # Did we get any results?
    if table is None or table.is_empty() or len(table.array) == 0:
        if table is None:
            # This indicates we could not connect to the server
            log.warning(f"Unable to connect to {server}")
        else:
            log.stdinfo(f"No results returned from {server}")
        return

    array = table.array

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
        array = array[array['mode'] == 1]

    # It turns out to be not viable to use UCDs to select the columns,
    # even for the id, ra, and dec. Even with vizier. <sigh>
    # The first column is our running integer column
    ret_table = Table([list(range(1, len(array)+1))],
                      names=('Id',), dtype=('i4',))

    ret_table.add_column(Column(array[svr.server_cols[0]], name='Cat_Id',
                                dtype='a'))
    ret_table.add_column(Column(array[svr.server_cols[1]], name='RAJ2000',
                                dtype='f8', unit='deg'))
    ret_table.add_column(Column(array[svr.server_cols[2]], name='DEJ2000',
                                dtype='f8', unit='deg'))

    # Now the photometry columns
    for col in range(3, len(svr.cat_cols)):
        ret_table.add_column(Column(array[svr.server_cols[col]], name=svr.cat_cols[col],
                                    dtype='f4', unit='mag', format='8.4f'))

    header = add_header_to_table(ret_table)
    header['CATALOG'] = (catalog.upper(), 'Origin of source catalog')

    # Add comments to the header to describe it
    header.add_comment(f'Source catalog derived from the {catalog} catalog')
    header.add_comment(f'Source catalog fetched from server at {svr.url}')
    header.add_comment(f'Delivered Table name from server:  {table.name}')
    for col in range(len(svr.cat_cols)):
        header.add_comment('UCD for field {} is {}'.format(
            svr.cat_cols[col], table.get_field_by_id(svr.server_cols[col]).ucd))
    return ret_table
