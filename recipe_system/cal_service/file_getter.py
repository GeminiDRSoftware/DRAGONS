import shutil
import os.path
import requests
from requests.exceptions import HTTPError
from requests.exceptions import Timeout
from requests.exceptions import ConnectionError

from ..config import globalConf
from gempy.utils.logutils import get_logger
from .calrequestlib import generate_md5_digest

# get_file_itterator is the only function in this module called externally
# (from calrequestlib.py)

def get_request(url, filename, calmd5):
    """
    This function was moved from calrequestlib.py as it belongs more naturally
    in this module as it is the only thing that calls methods in this module.
    Those methods where replaced with the CachedFileGetter class.
    :param url: URL to fetch data from
    :param filename: Filename to put data into
    :param calmd5: md5 hexdigest of cal we are requesting
    :return: None
    """
    cfg = CachedFileGetter()
    cfg.get_request(url, filename, calmd5)

class CachedFileGetter(object):
    """
    This class implements a file getter with an optional system- (or user-)
    wide cache. To enable the cache, set system_calcache_dir to point at the
    cache directory in the [calibs] section of your dragonsrc. If this is unset
    or None, no caching is attempted. This cache is only really useful on
    machines which run lots of instances of reduce in separate working
    directories, notably for archive processing.

    This class is not a singleton, and we don't store any cache concurrency data
    in the class, we check file existence and compute md5 digests every time we
    service a request.
    """
    def __init__(self):
        self.cachedir = None
        self.log = get_logger(__name__)

        calconf = globalConf['calibs']
        if calconf:
            self.cachedir = calconf.get('system_calcache_dir')

    def _fetchurltofile(self, url, filepath):
        """
        Fetch the contents of url and put them in a file at filepath
        :param url: url to fetch
        :param filepath: file to store contents in
        :return: None
        """
        with open(filepath, 'wb') as fp:
            r = requests.get(url, stream=True, timeout=10.0)
            try:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=None):
                    fp.write(chunk)
            except HTTPError as err:
                raise GetterError(
                    ["Could not retrieve {}".format(url), str(err)])
            except ConnectionError as err:
                raise GetterError(
                    ["Unable to connect to url {}".format(url), str(err)])
            except Timeout as terr:
                raise GetterError(["Request timed out", str(terr)])

    def get_request(self, url, filename, calmd5):
        # This replaces the old calrequestlib.get_request() function
        if url.startswith("file://"):
            # If the URL is a local file, there's no point caching it anyway.
            # Simply copy the file referenced in the URL to filename
            urlfilepath = url.split('://', 1)[1]
            shutil.copyfile(urlfilepath, filename)
            return

        if self.cachedir is None:
            # Cache is disabled, Call requests to fetch URL, write data
            # directly to filename
            self._fetchurltofile(url, filename)
            return

        # Cache is enabled
        # Is cachefilename in cache?
        cachefilename = os.path.join(self.cachedir, os.path.basename(filename))
        if os.path.exists(cachefilename):
            # Does it have the correct md5?
            cachemd5 = generate_md5_digest(cachefilename)
            if cachemd5 == calmd5:
                self.log.debug(f"Cache hit, copying {cachefilename} to "
                               f"{filename}")
                shutil.copyfile(cachefilename, filename)
                return filename
            else:
                self.log.debug(f"Cache miss - file exists but wrong md5: "
                               f"{cachemd5=}, {calmd5=}")
        else:
            self.log.debug(f"Cache miss - {cachefilename} not in cache")
        # Fetch the URL to the cache
        self.log.debug(f"Fetching {url} to cache file {cachefilename}")
        self._fetchurltofile(url, cachefilename)
        # And copy it out to the destination
        shutil.copyfile(cachefilename, filename)


class GetterError(Exception):
    def __init__(self, messages):
        self.messages = messages

