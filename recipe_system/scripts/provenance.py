
# parsing the command line
import sys
from optparse import OptionParser

import astrodata, gemini_instruments
from astrodata.provenance import provenance_summary


def parse_args():
    parser = OptionParser()
    parser.set_description("""'provenance' is a script to view a summary of the provenance in a given FITS file.
    """)
    parser.add_option("-p", "--provenance", dest="provenance", action="store_true",
                      default=True,
                      help="show the top-level provenance records")
    parser.add_option("--provenance_history", dest="history", action="store_true",
                      default=True,
                      help="show the provenance history records")

    (options, args) = parser.parse_args()

    # Show options if none selected
    if not args:
        parser.print_help()
        sys.exit()
    return options, args


if __name__ == "__main__":
    options, args = parse_args()
    for arg in args:
        try:
            ad = astrodata.open(arg)
            print(f"Reading Provenance for {arg}\n")
            print(provenance_summary(ad, provenance=options.provenance, provenance_history=options.history))
        except astrodata.AstroDataError:
            print(f"Unable to open {arg} with DRAGONS\n")
