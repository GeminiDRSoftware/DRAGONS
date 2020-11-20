from astropy.table import Table
from datetime import datetime


PROVENANCE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def add_provenance(ad, filename, md5, primitive, timestamp=None):
    """
    Add the given provenance entry to the full set of provenance records on
    this object.

    Provenance is not added if the incoming md5 is None or ''.  These indicate
    source data for the provenance that are not on disk, so not useful as
    provenance.

    Parameters
    ----------
    ad : `astrodata.AstroData`
    filename : str
    md5 : str
    primitive : str
    timestamp : `datetime.datetime`

    """
    if md5 is None or md5 == '':
        # not a real input, we can ignore this one
        return

    if timestamp is None:
        timestamp = datetime.now().strftime(PROVENANCE_DATE_FORMAT)

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


def add_provenance_history(ad, timestamp_start, timestamp_stop, primitive, args):
    """
    Add the given ProvenanceHistory entry to the full set of history records
    on this object.

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
    if hasattr(ad, 'PROVENANCE_HISTORY'):
        for row in ad.PROVENANCE_HISTORY:
            if timestamp_start == row[0] and \
                    timestamp_stop == row[1] and \
                    primitive == row[2] and \
                    args == row[3]:
                # already in the history, skip
                return

    colsize = len(args)+1
    if hasattr(ad, 'PROVENANCE_HISTORY'):
        colsize = max(colsize, max(len(ph[3]) for ph in ad.PROVENANCE_HISTORY) + 1)

        timestamp_start_arr = [ph[0] for ph in ad.PROVENANCE_HISTORY]
        timestamp_stop_arr = [ph[1] for ph in ad.PROVENANCE_HISTORY]
        primitive_arr = [ph[2] for ph in ad.PROVENANCE_HISTORY]
        args_arr = [ph[3] for ph in ad.PROVENANCE_HISTORY]
    else:
        timestamp_start_arr = []
        timestamp_stop_arr = []
        primitive_arr = []
        args_arr = []

    timestamp_start_arr.append(timestamp_start)
    timestamp_stop_arr.append(timestamp_stop)
    primitive_arr.append(primitive)
    args_arr.append(args)

    dtype = ("S28", "S28", "S128", "S%d" % colsize)
    ad.append(Table(
        [timestamp_start_arr, timestamp_stop_arr, primitive_arr, args_arr],
        names=('timestamp_start', 'timestamp_stop', 'primitive', 'args'),
        dtype=dtype), name="PROVENANCE_HISTORY")


def clone_provenance(provenance_data, ad):
    """
    For a single input's provenance, copy it into the output
    `AstroData` object as appropriate.

    This takes a dictionary with a source filename, md5 and both it's
    original provenance and provenance_history information.  It duplicates
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


def clone_provenance_history(provenance_history_data, ad):
    """
    For a single input's provenance history, copy it into the output
    `AstroData` object as appropriate.

    This takes a dictionary with a source filename, md5 and both it's
    original provenance and provenance_history information.  It duplicates
    the provenance data into the outgoing `AstroData` ad object.

    Parameters
    ----------
    provenance_history_data :
        pointer to the `AstroData` table with the history information.
        *Note* this may be the output `~astrodata.AstroData` as well, so we
        need to handle that.
    ad : `astrodata.AstroData`
        Outgoing `~astrodata.AstroData` object to add provenance history data
        to.

    """
    phd = [(prov_hist[0], prov_hist[1], prov_hist[2], prov_hist[3])
           for prov_hist in provenance_history_data]
    for ph in phd:
        add_provenance_history(ad, ph[0], ph[1], ph[2], ph[3])
