"""
This module provides some functions to regularize BPM files for DRAGONS
comsumption.

    bpmify(ad) -
      Fixes certain keywords for DRAGONS BPM files.

    dragons_bpm(ad) -
      creates a nominal DRAGONS BPM file name and writes this new filename.

    tabl(<filelist>) -
      Pass a list of bpmfiles, with directory prefix or or not.
      The function produces a table (stdout) of the BPMS (see table below).

The table below is created from a glob list of extant DRAGONS
compatible BPM files and names. This table is current as of 2020-05-05

>>> bpmfiles = glob.glob('*full*_v1.fits')
>>> bpmtab.tabl(bpmfiles)

  DIRECTORY /Users/kanderso/Gemini/GitHub/DRAGONS/geminidr/gmos/lookups/BPM/

	File				OBJ   BININNG  BITPIX    Shape 0         Det Name      Camera
======================
gmos-n_bpm_HAM_11_full_12amp_v1.fits	BPM     (1 1)	16	(4224, 544)	Hamamatsu-N	GMOS-N
gmos-n_bpm_HAM_12_full_12amp_v1.fits	BPM     (1 2)	16	(2112, 544)	Hamamatsu-N	GMOS-N
gmos-n_bpm_HAM_14_full_12amp_v1.fits	BPM     (1 4)	16	(1056, 544)	Hamamatsu-N	GMOS-N
gmos-n_bpm_HAM_21_full_12amp_v1.fits	BPM     (2 1)	16	(4224, 288)	Hamamatsu-N	GMOS-N
gmos-n_bpm_HAM_22_full_12amp_v1.fits	BPM     (2 2)	16	(2112, 288)	Hamamatsu-N	GMOS-N
gmos-n_bpm_HAM_24_full_12amp_v1.fits	BPM     (2 4)	16	(1056, 288)	Hamamatsu-N	GMOS-N
gmos-n_bpm_HAM_41_full_12amp_v1.fits	BPM     (4 1)	16	(4224, 160)	Hamamatsu-N	GMOS-N
gmos-n_bpm_HAM_42_full_12amp_v1.fits	BPM     (4 2)	16	(2112, 160)	Hamamatsu-N	GMOS-N
gmos-n_bpm_HAM_44_full_12amp_v1.fits	BPM     (4 4)	16	(1056, 160)	Hamamatsu-N	GMOS-N
gmos-s_bpm_EEV_11_full_3amp_v1.fits	BPM     (1 1)	16	(4608, 2048)	EEV	GMOS-S
gmos-s_bpm_EEV_12_full_3amp_v1.fits	BPM     (1 2)	16	(2304, 2048)	EEV	GMOS-S
gmos-s_bpm_EEV_14_full_3amp_v1.fits	BPM     (1 4)	16	(1152, 2048)	EEV	GMOS-S
gmos-s_bpm_EEV_21_full_3amp_v1.fits	BPM     (2 1)	16	(4608, 1024)	EEV	GMOS-S
gmos-s_bpm_EEV_22_full_3amp_v1.fits	BPM     (2 2)	16	(2304, 1024)	EEV	GMOS-S
gmos-s_bpm_EEV_24_full_3amp_v1.fits	BPM     (2 4)	16	(1152, 1024)	EEV	GMOS-S
gmos-s_bpm_EEV_41_full_3amp_v1.fits	BPM     (4 1)	16	(4608, 512)	EEV	GMOS-S
gmos-s_bpm_EEV_42_full_3amp_v1.fits	BPM     (4 2)	16	(2304, 512)	EEV	GMOS-S
gmos-s_bpm_EEV_44_full_3amp_v1.fits	BPM     (4 4)	16	(1152, 512)	EEV	GMOS-S
======================

Readers will note the messy OBJ names rather than 'BPM', the DRAGONS standard
OBJECT name for BPM files. These "noisey" OBJECT names are deprecated and
DRAGONS BPMs shall use 'BPM' as OBJECT value.

"""




import os
import datetime

import numpy as np

import astrodata
import gemini_instruments

# -----------------------
def bpmify(ad):
    delta = False
    if not ad.phu.get('BPMASK'):
        ad.phu.set('BPMASK', datetime.datetime.now().isoformat()[:-7])
        print("Set BPM keyword: 'BPMASK'")
        delta = True

    if ad.object() != 'BPM':
        ad.phu.set('OBJECT', 'BPM')
        print("Set keyword: 'OBJECT' to 'BPM'")
        delta = True

    for ext in ad:
        if ext.data.dtype != np.uint16:
            ext.data = ext.data.astype(np.uint16)
            delta = True

    if delta:
        ad.write(filename="interim_"+ad.filename)
    return

def dragons_bpm(ad, prefix=None):
    """
    Add header things to a BPM to convert it into a DRAGONS compatible
    BPM. Sets header keywords,

    OBJECT
    BPMASK

    Writes out a file with the DRAGONS naming convention.

    """
    if prefix is None:
        prefix = ''
    namps = None
    nbuffer = prefix+"{}_{}_{}_{}_{}_{}_{}.fits"

    if ad.detector_name(pretty=True).upper()[:3] == 'EEV':
        namps = '3amp'
    elif ad.detector_name(pretty=True).upper()[:3] == 'e2vDD':
        namps = '6amp'
    elif ad.detector_name(pretty=True).upper()[:3] == 'HAM':
        namps = '12amp'
    else:
        raise TypeError("Unrecognized detector name")

    new_bpm_name = nbuffer.format(
        ad.camera().lower(),
        ad.object().lower(),
        ad.detector_name(pretty=True).upper()[:3],
        str(ad.detector_x_bin())+str(ad.detector_y_bin()),
        ad.detector_roi_setting()[:4].lower(),
        namps,
        "v1"
    )
    try:
        ad.write(filename=new_bpm_name)
    except OSError as err:
        print("Warning: filename is over subscribed. Higher version used.")
        ad.write(filename=new_bpm_name, overwrite=True)

    return new_bpm_name


def tabl(ffiles):
    dirheader = "    \n\n  DIRECTORY {}/\n "
    header = "\tFile\t\t\t\tOBJ   BININNG  BITPIX    Shape 0         Det Name"
    header += "      Camera\n" + "="*22
    rowl = "{}\t{}     ({} {})\t{}\t{}\t{}\t{}"
    rows = "{}\t{}     ({} {})\t{}\t{}\t{}\t{}"
    dirf,ff = os.path.split(ffiles[0])
    if not dirf:
        dirf = '.'

    print(dirheader.format(dirf))
    print(header)
    ffiles.sort()
    for ff in ffiles:
        ad = astrodata.open(ff)
        fname = os.path.split(ff)[-1]
        if len(fname) < 25:
            print(rows.format(
                fname,
                ad.object(),
                ad.detector_x_bin(), ad.detector_y_bin(),
                ad[0].hdr['BITPIX'],
                ad[0].data.shape,
                ad.detector_name(pretty=True),
                ad.camera())
            )
        else:
            print(rowl.format(
                fname,
                ad.object(),
                ad.detector_x_bin(), ad.detector_y_bin(),
                ad[0].hdr['BITPIX'],
                ad[0].data.shape,
                ad.detector_name(pretty=True),
                ad.camera())
            )
    print("="*22)
    return
