#
#                                                                 calurl_dict.py
# ------------------------------------------------------------------------------
# FITS Store URLs, etc

calurl_dict = {
    "CALMGR"       : "http://fits/calmgr",
    "LOCALCALMGR"  : "http://localhost:%(httpport)d/calmgr/%(caltype)s",
    "UPLOADPROCCAL": "http://fits/upload_processed_cal",
    "UPLOADCOOKIE" : "qap_upload_processed_cal_ok",
    "QAMETRICURL"  : "http://fits/qareport",
    "QAQUERYURL"   : "http://fits/qaforgui",
}
