#
#                                                                     QAP Gemini
#
#                                                                 calurl_dict.py
#                                                                        07-2013
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
# FITS Store URLs, etc

calurl_dict = {
     "CALMGR" : "http://fits/calmgr",
"LOCALCALMGR" : "http://localhost:%(httpport)d/calmgr/%(caltype)s",
#"UPLOADPROCCAL": "http://hbffits3.hi.gemini.edu/upload_processed_cal",
"UPLOADPROCCAL": "http://fits/upload_processed_cal",
"QAMETRICURL" : "http://fits/qareport",
"QAQUERYURL"  : "http://fits/qaforgui",
#"QAMETRICURL" : "http://cpofits1new/qareport",   # test site
#"QAQUERYURL"  : "http://cpofits1new/qaforgui"    # test site
              }
