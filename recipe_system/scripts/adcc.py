#!/usr/bin/env python
#                                                                     QAP Gemini
#
#                                                                        adcc.py
# ------------------------------------------------------------------------------
"""
Automated Dataflow Coordination Center

"""
import sys

from argparse import ArgumentParser

from recipe_system import __version__

from recipe_system.adcc.adcclib import ADCC
# ------------------------------------------------------------------------------
def buildArgs():
    parser = ArgumentParser(description="Automated Data Communication Center "
                            "(ADCC), v{}".format(__version__))

    parser.add_argument("-d", "--dark", dest="dark", action="store_true",
                        help="Use the adcc faceplate 'dark' theme.")

    parser.add_argument("-v","--verbose", dest="verbosity", action="store_true",
                        help="increase HTTP client messaging on GET requests.")

    parser.add_argument("--startup-report", dest="adccsrn", default="adccReport",
                        help = "file name for adcc startup report.")

    parser.add_argument("--http-port", dest="httpport", default=8777, type=int,
                        help="Response port for the web interface. "
                        "i.e. http://localhost:<http-port>. "
                        "Default is 8777.")

    args = parser.parse_args()
    return args

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    adcc = ADCC(buildArgs())
    sys.exit(adcc.main())
