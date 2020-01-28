

class Provenance(object):
    """
    Tracks the set of actual files that were involved in producing this one.

    This is used to track the set of files (bias, flats, etc) that were used in the processing to produce a file.
    Provenance data is typically stored in the output file.
    """

    def __init__(self, timestamp, filename, md5, provenance_added_by):
        self.timestamp = timestamp
        self.filename = filename
        self.md5 = md5
        self.provenance_added_by = provenance_added_by

    def __repr__(self):
        timestamp_str = self.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')
        return '{"timestamp": "%s", "filename": "%s", "md5": "%s", "provenance_added_by": "%s"}' \
            % (timestamp_str, self.filename, self.md5, self.provenance_added_by)

    def __eq__(self, other):
        if isinstance(other, Provenance):
            if other.timestamp == self.timestamp \
                and other.filename == self.filename \
                and other.md5 == self.md5 \
                and other.provenance_added_by == self.provenance_added_by:
                    return True
        return False

    def __hash__(self):
        return self.timestamp.__hash__()


class ProvenanceHistory(object):
    """
    Tracks the set of primitive operations that were involved in producing this data.

    This is used to track the set of primitives that were used during processing to produce a file.
    ProvenanceHistoryData is typicaly stored in the output file.
    """
    def __init__(self, timestamp_start, timestamp_stop, primitive, args):
        self.timestamp_start = timestamp_start
        self.timestamp_stop = timestamp_stop
        self.primitive = primitive
        self.args = args

    def __repr__(self):
        timestamp_start_str = self.timestamp_start.strftime('%Y-%m-%d %H:%M:%S.%f')
        timestamp_stop_str = self.timestamp_stop.strftime('%Y-%m-%d %H:%M:%S.%f')
        return '{"timestamp_start": "%s", "timestamp_stop": "%s", "primitive": "%s", "args": "%s"}' \
            % (timestamp_start_str, timestamp_stop_str, self.primitive, self.args)

    def __eq__(self, other):
        if isinstance(other, ProvenanceHistory):
            if other.timestamp_start == self.timestamp_start \
                and other.timestamp_stop == self.timestamp_stop \
                and other.primitive == self.primitive \
                and other.args == self.args:
                return True
        return False

    def __hash__(self):
        return self.timestamp_start.__hash__()
