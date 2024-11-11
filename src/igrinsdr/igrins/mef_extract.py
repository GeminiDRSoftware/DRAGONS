#!/usr/bin/env python3

import sys
import argparse
import os
import glob
from astropy.io import fits
from pathlib import Path

def unbundle(indir: Path, utdate: str, outdir: Path):
    fn_list = sorted(indir.glob(f"N{utdate}*.fits*")) # ".fits*" to include gz, bz2 files

    if len(fn_list) == 0:
        print("no matching files are found")
        return

    outdir.mkdir(parents=True, exist_ok=True)

    for fi_ in fn_list:
        fi = fi_.name
        obsdate = fi[1:9]
        obsid = fi[10:14]

        hdu = fits.open(indir / fi)

        hd0 = hdu[0].header.copy()
        if hd0['OBSCLASS'] != 'acq':
            del hd0["EXTEND"]
            hd0.set("NAXIS1",0,after="NAXIS")
            hd0.set("NAXIS2",0,after="NAXIS1")

            band = ["H","K"]

            for i in range(2):
                hd = hd0.copy()
                hd.update(hdu[i+1].header[1:])
                im = hdu[i+1].data

                filename = f"SDC{band[i]}_{obsdate}_{obsid}.fits"
                hd["ORIGNAME"] = filename
                fits.PrimaryHDU(header=hd,data=im).writeto(
                    outdir / filename, overwrite=True)


def main():
    descriptions = """Given the ut_date, read the MEFs files of that date ('N{ut_date}*.fits')
and extract H & K spectra extensions into directory of ./indata/{ut_date} with names like
SDCH_{ut_date}_*.fits, SDCK_{ut_date}_*.fits.
"""
    parser = argparse.ArgumentParser(
        # formatter_class=argparse.RawTextHelpFormatter,
        description=descriptions,
        epilog="examples:\n" +
        "  mef_extract.py 20240425\n\n")

    parser.add_argument("ut_date", help="UT Date of Data")
    parser.add_argument("--mefdir", help="Directory containing MEF files.",
                        default=".")
    parser.add_argument("--outdir", type=str, help="Ouput data directory. Default is ./indata/{ut_date}",
                        default="./indata/{ut_date}")
    # main(parser.parse_args(args=None if sys.argv[1:] else ["--help"]))

    args = parser.parse_args(args=None if sys.argv[1:] else ["--help"])

    outdir = args.outdir.format(ut_date=args.ut_date)

    unbundle(Path(args.mefdir), args.ut_date, Path(outdir))


if __name__ == "__main__":
    main()
