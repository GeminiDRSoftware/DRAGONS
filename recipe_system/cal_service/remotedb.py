# Defines the RemoteDB class for calibration returns. This is a high-level
# interface to FITSstore. It may be subclassed in future

from os.path import basename

from .caldb import CalDB, CalReturn, cascade
from .calrequestlib import get_cal_requests, generate_md5_digest
from .calurl_dict import calurl_dict
from .transport_request import upload_calibration


class RemoteDB(CalDB):
    def __init__(self, name=None, valid_caltypes=None, autostore=False,
                 log=None):
        super().__init__(name=name, autostore=autostore, log=log,
                         valid_caltypes=valid_caltypes)

    def _get_calibrations(self, adinputs, caltype=None, procmode=None):
        cal_requests = get_cal_requests(adinputs, caltype, procmode=procmode,
                                        is_local=False)
        cals = []
        for rq in cal_requests:
            pass

    @cascade
    def _store_calibrations(self, calfiles, caltype=None):
        if self.autostore:
            for calfile in calfiles:
                upload_calibration(calfile)
