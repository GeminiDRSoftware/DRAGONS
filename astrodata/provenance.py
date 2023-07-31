import json

from astropy.table import Table
from datetime import datetime


def add_provenance(ad, filename, md5, primitive, timestamp=None):
    """
    Add the given provenance entry to the full set of provenance records on
    this object.

    Provenance is added even if the incoming md5 is None or ''.  This indicates
    source data for the provenance that are not on disk.

    Parameters
    ----------
    ad : `astrodata.AstroData`
    filename : str
    md5 : str
    primitive : str
    timestamp : `datetime.datetime`

    """
    # Handle data where the md5 is None. It will need to be a string type to
    # store in the FITS table.
    md5 = '' if md5 is None else md5

    if timestamp is None:
        timestamp = datetime.utcnow().isoformat()

    if hasattr(ad, 'PROVENANCE'):
        existing_provenance = ad.PROVENANCE
        for row in existing_provenance:
            if row[1] == filename and \
                    row[2] == md5 and \
                    row[3] == primitive:
                # nothing needed, we already have it
                return

    if not hasattr(ad, 'PROVENANCE'):
        timestamp_data = [timestamp]
        filename_data = [filename]
        md5_data = [md5]
        provenance_added_by_data = [primitive]
        ad.PROVENANCE = Table(
            [timestamp_data, filename_data, md5_data, provenance_added_by_data],
            names=('timestamp', 'filename', 'md5', 'provenance_added_by'),
            dtype=('S28', 'S128', 'S128', 'S128'))
    else:
        provenance = ad.PROVENANCE
        provenance.add_row((timestamp, filename, md5, primitive))


def add_history(ad, timestamp_start, timestamp_stop, primitive, args):
    """
    Add the given History entry to the full set of history records on this
    object.

    Parameters
    ----------
    ad : `astrodata.AstroData`
        AstroData object to add history record to.
    timestamp_start : `datetime.datetime`
        Date of the start of this operation.
    timestamp_stop : `datetime.datetime`
        Date of the end of this operation.
    primitive : str
        Name of the primitive performed.
    args : str
        Arguments used for the primitive call.

    """
    # If the ad instance has the old 'PROVHISTORY' extenstion name, rename it
    # now to 'HISTORY'
    if hasattr(ad, 'PROVHISTORY'):
        ad.HISTORY = ad.PROVHISTORY
        del ad.PROVHISTORY

    # I modified these indices, so making this method adaptive to existing
    # histories with the old ordering.  This also makes modifying the order
    # in future easier
    primitive_col_idx, args_col_idx, timestamp_start_col_idx, \
        timestamp_stop_col_idx = find_history_column_indices(ad)

    if hasattr(ad, 'HISTORY') and None not in (primitive_col_idx, args_col_idx,
                                               timestamp_stop_col_idx,
                                               timestamp_start_col_idx):
        for row in ad.HISTORY:
            if timestamp_start == row[timestamp_start_col_idx] and \
                    timestamp_stop == row[timestamp_stop_col_idx] and \
                    primitive == row[primitive_col_idx] and \
                    args == row[args_col_idx]:
                # already in the history, skip
                return

    colsize = len(args)+1
    if hasattr(ad, 'HISTORY'):
        colsize = max(colsize, (max(len(ph[args_col_idx])
                                    for ph in ad.HISTORY) + 1)
        if args_col_idx is not None else 16)

        timestamp_start_arr = [ph[timestamp_start_col_idx]
                               if timestamp_start_col_idx is not None else ''
                               for ph in ad.HISTORY]
        timestamp_stop_arr = [ph[timestamp_stop_col_idx]
                              if timestamp_stop_col_idx is not None else ''
                              for ph in ad.HISTORY]
        primitive_arr = [ph[primitive_col_idx]
                         if primitive_col_idx is not None else ''
                         for ph in ad.HISTORY]
        args_arr = [ph[args_col_idx] if args_col_idx is not None else ''
                    for ph in ad.HISTORY]
    else:
        timestamp_start_arr = []
        timestamp_stop_arr = []
        primitive_arr = []
        args_arr = []

    timestamp_start_arr.append(timestamp_start)
    timestamp_stop_arr.append(timestamp_stop)
    primitive_arr.append(primitive)
    args_arr.append(args)

    dtype = ("S128", "S%d" % colsize, "S28", "S28")
    ad.HISTORY = Table([primitive_arr, args_arr, timestamp_start_arr,
                        timestamp_stop_arr],
                       names=('primitive', 'args', 'timestamp_start',
                              'timestamp_stop'),
                       dtype=dtype)


def clone_provenance(provenance_data, ad):
    """
    For a single input's provenance, copy it into the output
    `AstroData` object as appropriate.

    This takes a dictionary with a source filename, md5 and both its
    original provenance and history information.  It duplicates
    the provenance data into the outgoing `AstroData` ad object.

    Parameters
    ----------
    provenance_data :
        Pointer to the `~astrodata.AstroData` table with the provenance
        information.  *Note* this may be the output `~astrodata.AstroData`
        as well, so we need to handle that.
    ad : `astrodata.AstroData`
        Outgoing `~astrodata.AstroData` object to add provenance data to.


    """
    pd = [(prov[1], prov[2], prov[3], prov[0]) for prov in provenance_data]
    for p in pd:
        add_provenance(ad, p[0], p[1], p[2], timestamp=p[3])


def clone_history(history_data, ad):
    """
    For a single input's history, copy it into the output
    `AstroData` object as appropriate.

    This takes a dictionary with a source filename, md5 and both its
    original provenance and history information.  It duplicates
    the history data into the outgoing `AstroData` ad object.

    Parameters
    ----------
    history_data :
        pointer to the `AstroData` table with the history information.
        *Note* this may be the output `~astrodata.AstroData` as well, so we
        need to handle that.
    ad : `astrodata.AstroData`
        Outgoing `~astrodata.AstroData` object to add history data
        to.

    """
    primitive_col_idx, args_col_idx, timestamp_start_col_idx, \
        timestamp_stop_col_idx = find_history_column_indices(ad)
    hd = [(hist[timestamp_start_col_idx], hist[timestamp_stop_col_idx],
            hist[primitive_col_idx], hist[args_col_idx])
           for hist in history_data]
    for h in hd:
        add_history(ad, h[0], h[1], h[2], h[3])


def find_history_column_indices(ad):
    if hasattr(ad, 'HISTORY'):
        primitive_col_idx = None
        args_col_idx = None
        timestamp_start_col_idx = None
        timestamp_stop_col_idx = None
        for idx, colname in enumerate(ad.HISTORY.colnames):
            if colname == 'primitive':
                primitive_col_idx = idx
            elif colname == 'args':
                args_col_idx = idx
            elif colname == 'timestamp_start':
                timestamp_start_col_idx = idx
            elif colname == 'timestamp_stop':
                timestamp_stop_col_idx = idx
    else:
        # defaults
        primitive_col_idx = 0
        args_col_idx = 1
        timestamp_start_col_idx = 2
        timestamp_stop_col_idx = 3

    return primitive_col_idx, args_col_idx, timestamp_start_col_idx, \
        timestamp_stop_col_idx


def provenance_summary(ad, provenance=True, history=True):
    """
    Generate a pretty text display of the provenance information for an
    `~astrodata.core.AstroData`.

    This pulls the provenance and history information from a
    `~astrodata.core.AstroData` object and formats it for readability.  The
    primitive arguments in the history are wrapped across multiple lines to
    keep the overall width manageable.

    Parameters
    ----------
    ad : :class:`~astrodata.core.AstroData`
        Input data to read provenance from
    provenance : bool
        True to show provenance
    history : bool
        True to show the history with associated parameters and timestamps

    Returns
    -------
        str representation of the provenance and history
    """
    retval = ""
    if provenance:
        if hasattr(ad, 'PROVENANCE'):
            retval = f"Provenance\n----------\n{ad.PROVENANCE}\n"
        else:
            retval = "No Provenance found\n"
    if history:
        if provenance:
            retval += "\n"  # extra blank line between
        if hasattr(ad, 'PROVHISTORY'):
            retval += "Warning: File uses old PROVHISTORY extname."
            ad.HISTORY = ad.PROVHISTORY
        if hasattr(ad, 'HISTORY'):
            retval += "History\n-------\n"

            primitive_col_idx, args_col_idx, timestamp_start_col_idx, \
                timestamp_stop_col_idx = find_history_column_indices(ad)

            primitive_col_size = 9
            timestamp_start_col_size = 28
            timestamp_stop_col_size = 28
            args_col_size = 16

            # infer args size by finding the max for the folded json values
            for row in ad.HISTORY:
                argsstr = row[args_col_idx]
                args = json.loads(argsstr)
                argspp = json.dumps(args, indent=4)
                for line in argspp.split('\n'):
                    args_col_size = max(args_col_size, len(line))
                primitive_col_size = max(primitive_col_size,
                                         len(row[primitive_col_idx]))

            # Titles
            retval += f'{"Primitive":<{primitive_col_size}} ' \
                      f'{"Args":<{args_col_size}} ' \
                      f'{"Start":<{timestamp_start_col_size}} {"Stop"}\n'
            # now the lines
            retval += f'{"":{"-"}<{primitive_col_size}} ' \
                      f'{"":{"-"}<{args_col_size}} ' \
                      f'{"":{"-"}<{timestamp_start_col_size}} ' \
                      f'{"":{"-"}<{timestamp_stop_col_size}}\n'

            # Rows, looping over args lines
            for row in ad.HISTORY:
                primitive = row[primitive_col_idx]
                args = row[args_col_idx]
                start = row[timestamp_start_col_idx]
                stop = row[timestamp_stop_col_idx]
                first = True
                try:
                    parseargs = json.loads(args)
                    args = json.dumps(parseargs, indent=4)
                except Exception:
                    pass  # ok, just use whatever non-json was in there
                for argrow in args.split('\n'):
                    if first:
                        retval += f'{primitive:<{primitive_col_size}} ' \
                                  f'{argrow:<{args_col_size}} ' \
                                  f'{start:<{timestamp_start_col_size}} ' \
                                  f'{stop}\n'
                    else:
                        retval += f'{"":<{primitive_col_size}} {argrow}\n'
                    # prep for additional arg rows without duplicating the
                    # other values
                    first = False
        else:
            retval += "No Provenance History found.\n"
    return retval
