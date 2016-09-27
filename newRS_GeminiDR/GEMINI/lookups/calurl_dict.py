#
#                                                                     QAP Gemini
#
#                                                                 calurl_dict.py
#                                                                        07-2013
# ------------------------------------------------------------------------------
# $Id: calurl_dict.py 5331 2015-09-14 23:48:57Z phirst $
# ------------------------------------------------------------------------------
__version__      = '$Revision: 5331 $'[11:-2]
__version_date__ = '$Date: 2015-09-14 20:48:57 -0300 (Mon, 14 Sep 2015) $'[7:-2]
# ------------------------------------------------------------------------------
# FITS Store URLs, etc

calurl_dict = {
     "CALMGR" : "http://fits/calmgr",
"LOCALCALMGR" : "http://localhost:%(httpport)d/calmgr/%(caltype)s",
#"UPLOADPROCCAL": "http://hbffits3.hi.gemini.edu/upload_processed_cal",
"UPLOADPROCCAL": "http://fits/upload_processed_cal",
"UPLOADCOOKIE": "qap_upload_processed_cal_ok",
"QAMETRICURL" : "http://fits/qareport",
"QAQUERYURL"  : "http://fits/qaforgui",
#"QAMETRICURL" : "http://cpofits1new/qareport",   # test site
#"QAQUERYURL"  : "http://cpofits1new/qaforgui"    # test site
              }
