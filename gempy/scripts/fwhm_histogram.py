#!/usr/bin/env python
'''
This script takes a file with an OBJCAT in it, and displays an FWHM histogram
that it useful for asessing the validity of a QAP FWHM measurement.
It re-implements the algorithm used in gempy/gemini_tools.py clip_sources()
and to some degree the measureIQ primitive
- it could (and maybe should) be adapted to call clip_sources() directly

This was developed as a quick analysis tool to asess QAP FWHM source selection
and to understand comparisons with certain people's imexam results, but I'm sure
it would be useful for the DA/SOSs.
It can be run on any file with a sextractor style OBJCAT in it, for example
the _forStack files output by the QAP

Paul Hirst 20120321
'''

import sys
import astrodata
import gemini_instruments
import numpy as np
import matplotlib.pyplot as plt

def main():
    filename = sys.argv[1]

    ad = astrodata.from_file(filename)
    objcat = ad[0].OBJCAT

    fwhm_arcsec = objcat.field("FWHM_WORLD") * 3600.
    flux = objcat.field("FLUX_AUTO")
    fluxerr = objcat.field("FLUXERR_AUTO")
    ellip = objcat.field("ELLIPTICITY")
    sxflag = objcat.field("FLAGS")
    dqflag = objcat.field("IMAFLAGS_ISO")
    class_star = objcat.field("CLASS_STAR")
    area = objcat.field("ISOAREA_IMAGE")

    # Source is good if ellipticity defined and <0.5
    eflag = np.where((ellip > 0.5) | (ellip == -999), 1, 0)

    # Source is good if probability of being a star >0.9
    sflag = np.where(class_star < 0.9, 1, 0)

    # Source is good if isoarea < 20 pixels
    aflag = np.where(area < 20, 1, 0)

    # Source is good if better than 50:1 signal to noise
    snflag = np.where(flux < 50 * fluxerr, 1, 0)


    # Jump ahead a bit to calculate the clipped sources
    flags = sxflag | dqflag | sflag | eflag | aflag | snflag
    good = (flags == 0)
    records = np.rec.fromarrays([fwhm_arcsec[good]], names=["fwhm_arcsec"])
    fwhm = records["fwhm_arcsec"]
    mean = fwhm.mean()
    std = fwhm.std()
    upper = mean + std
    lower = mean - std
    clipflag = np.where((fwhm_arcsec > upper) | (fwhm_arcsec < lower), 1, 0)
    # Now apply the clipping and re-calculate the mean and sigma
    flags = flags | clipflag
    good = (flags == 0)
    records = np.rec.fromarrays([fwhm_arcsec[good]], names=["fwhm_arcsec"])
    fwhm = records["fwhm_arcsec"]
    mean = fwhm.mean()
    std = fwhm.std()
    upper = mean + std
    lower = mean - std
    print("FWHM: %.2f +- %.2f" % (mean, std))

    data = []
    labels = ('all', 'stars', 'hsnstars', 'clipped')
    flags = [sxflag | dqflag,
             sxflag | dqflag | sflag | eflag | aflag,
             sxflag | dqflag | sflag | eflag | snflag | aflag,
             sxflag | dqflag | sflag | eflag | snflag | aflag | clipflag]

    bins = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
            1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4)


    for i in range(len(flags)):

        # Use flag=0 to find good data
        good = (flags[i] == 0)
        rec = np.rec.fromarrays([fwhm_arcsec[good]], names=["fwhm_arcsec"])

        data.append(rec['fwhm_arcsec'])

    plt.xlabel('FWHM')
    plt.ylabel('Nstars')
    plt.hist(data, bins=bins, histtype='stepfilled', label=labels)
    plt.legend()
    plt.axvspan(lower, upper, 0.98, 1.0)
    plt.axvspan(mean - 0.01, mean + 0.01, 0.95, 1.0)
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
