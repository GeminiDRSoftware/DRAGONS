import requests
from requests.exceptions import HTTPError
from requests.exceptions import Timeout
from requests.exceptions import ConnectionError

from gempy.utils import logutils

class GetterError(Exception):
    def __init__(self, messages):
        self.messages = messages

def requests_getter(url):
    r = requests.get(url, timeout=10.0)
    try:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=128):
            yield chunk
    except HTTPError as err:
        raise GetterError(["Could not retrieve {}".format(url), str(err)])
    except ConnectionError as err:
        raise GetterError(["Unable to connect to url {}".format(url), str(err)])
    except Timeout as terr:
        raise GetterError(["Request timed out", str(terr)])

def plain_file_getter(url):
    path = url.split('://', 1)[1]
    try:
        with open(path) as source:
            while True:
                data = source.read(128)
                if not data:
                    break
                yield data
    except Exception as err:
        raise GetterError(["Problem accessing to {}".format(url), str(err)])

schema_mapper = {
    'http': requests_getter,
    'https': requests_getter,
    'ftp': requests_getter,
    'file': plain_file_getter
    }

def get_file_iterator(url):
    getter = schema_mapper[url.split('://', 1)[0]]
    return getter(url)
