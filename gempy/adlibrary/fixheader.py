DTYPES = {'int': int, 'float': float, 'str': str}


def modify_header(ad, extid=None, keyword=None, value=None, add=False,
                  dtype=None, logfn=print):
    """
    Modify one or more headers of an AstroData object

    Parameters
    ----------
    ad: AstroData
        the AD object to be modfied
    extid: int/""/None
        extensions to change (""=all, None=PHU)
    keyword: str
        header keyword to update/add
    value: float/int/str
        new value for keyword
    add: bool
        keyword needs to be added?
    dtype: "float"/"int"/"str"/None
        datatype of keyword if uncertain (None => use int or float if possible)
    logfn: callable
        function to use for logging

    Returns
    -------
    AstroData: modified AD object
    """
    def coerce(value, dtype):
        if dtype is not None:
            if isinstance(dtype, str):
                dtype = DTYPES[dtype]
            return dtype(value)
        try:
            v = int(value)
        except ValueError:
            try:
                v = float(value)
            except ValueError:
                return value
            else:
                return v
        return v

    if extid is None:  # PHU
        if keyword in ad.phu or add:
            if keyword in ad.phu:
                if dtype is None:
                    dtype = type(ad.phu[keyword])
                logfn(f"{ad.filename}: Updating {keyword}="
                      f"{ad.phu[keyword]} -> {value} in PHU")
            else:
                logfn(f"{ad.filename}: Adding {keyword}={value} in PHU")
            ad.phu[keyword] = coerce(value, dtype)
        elif ad.hdr.get(keyword).count(None) == 0:  # try all extensions
            extid = ''
        else:
            raise KeyError(f"{ad.filename}: {keyword} is not present "
                           "in all extensions")
    elif extid:  # one header only
        try:
            index = [ext.id for ext in ad].index(int(extid))
        except ValueError:
            raise ValueError(f"{ad.filename}: {extid} not a valid extension id")
        if keyword in ad[index].hdr or add:
            if keyword in ad[index].hdr:
                if dtype is None:
                    dtype = type(ad[index].hdr[keyword])
                logfn(f"{ad.filename}: Updating {keyword}="
                      f"{ad[index].hdr[keyword]} -> {value} "
                      f"in extension {extid}")
            else:
                logfn(f"{ad.filename}: Adding {keyword}={value} "
                      f"in extension {extid}")

            ad[index].hdr[keyword] = coerce(value, dtype)
        else:
            raise KeyError(f"{keyword} not found in {ad.filename}:{extid} and "
                           "'add' not selected")
    if extid == '':  # all headers
        already_values = ad.hdr.get(keyword)
        if 0 < already_values.count(None) < len(ad) and not add:
            logfn(f"{ad.filename}: {keyword} exists only in extensions " +
                  ", ".join([str(ext.id) for ext in ad if keyword in ext.hdr]))
        if not add:
            for ext, already in zip(ad, already_values):
                if already is not None:
                    logfn(f"{ad.filename}: Updating {keyword}="
                          f"{ext.hdr[keyword]} -> {value} "
                          f"in extension {ext.id}")
                    ext.hdr[keyword] = coerce(value, type(ext.hdr[keyword])
                                              if dtype is None else dtype)
        else:
            if dtype is not None:
                ad.hdr[keyword] = coerce(value, dtype)
            else:
                dtypes = set([type(ext.hdr.get(keyword)) for ext in ad]) - {type(None)}
                if len(dtypes) > 1:
                    raise ValueError(f"{keyword} does not have a unique datatype; "
                                     "please specify on the command line")
                elif not dtypes:
                    dtypes = [None]
                ad.hdr[keyword] = coerce(value, dtypes.pop())
            logfn(f"{ad.filename}: Adding {keyword}={value} in all extensions")
