from json import JSONEncoder, dumps, loads

import numpy as np
from astropy.table import Table

import math

# NanConvert from https://stackoverflow.com/questions/28639953/python-json-encoder-convert-nans-to-null-instead

def nan2None(obj):
    if isinstance(obj, dict):
        return {k:nan2None(v) for k,v in obj.items()}
    elif isinstance(obj, list):
        return [nan2None(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj

class NanConverter(JSONEncoder):
    def encode(self, obj, *args, **kwargs):
        return super().encode(nan2None(obj), *args, **kwargs)


def encode_array(obj):
    if hasattr(obj, "to_json"):
        return obj.to_json()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "dtype"):
        return obj.item()
    # # check if numpy polynomial. Need to be improved
    # elif hasattr(obj, "convert"):
    #     p = obj.convert(kind=P.Polynomial)
    #     return ["polynomial", p.coef]
    else:
        raise TypeError(repr(obj) + " is not JSON serializable")

def json_dumps(obj, *kl, **kw):
    if "default" not in kw:
        kw["default"] = encode_array

    kw["cls"] = NanConverter
    return dumps(obj, *kl, **kw)

def dict_to_table(j):

    keys, encoder_names, encoded = [], [], []
    for k, v in j.items():
        if isinstance(v, str):
            encoder_name, encoder = "", lambda o: o
        else:
            # encoder_name, encoder = "json", json_encoder
            encoder_name, encoder = "json", json_dumps
        keys.append(k)
        encoder_names.append(encoder_name)
        encoded.append(encoder(v))

    tbl = Table([keys, encoder_names, encoded], names=["key", "encoder_name", "encoded"])
    return tbl


def table_to_dict(tbl):
    j = {}
    for row in tbl:
        if row["encoder_name"] == "json":
            j[row["key"]] = loads(row["encoded"])

    return j
