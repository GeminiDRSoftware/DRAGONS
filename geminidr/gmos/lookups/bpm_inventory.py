"""
This module provides some functions to regularize BPM files for DRAGONS comsumption.

    bpmify(ad) -- Fixes certain keywords for DRAGONS BPM files.
    dragons_bpm(ad) -- creates a nominal DRAGONS BPM file name and writes this new filename.
    tabl() -- creates the table below, from a list of bpmfiles, with directory prefix or
              or not. All BPMS should be listed from the same directory.

For example, the table below is created from a glob list of BPM/*.fits, whence this
module exists (currently, $DRAGONS/geminidr/gmos/lookups)

>>> bpmfiles = glob.glob('BPM/*.fits')
>>> bpm_inventory.tabl(bpmfiles)

Readers will note the messy OBJ names rather than 'BPM', the DRAGONS standard OBJECT name for
BPM files. These "noisey" OBJECT names are deprecated and the BPM updated to verison, "_v3".
All _v3 editions of BPM files are corrected version that fixed OBJECT value and/or added the 
'BPMASK' keyword that signals an Astrodata tag of 'BPM'.

  DIRECTORY BPM/
 
	File			OBJ   BININNG  BITPIX    Shape 0         Det Name      Camera
======================
gmos-n_bpm_EEV_11_3amp_v1.fits	BPM     (1 1)	16	(4608, 2048)	EEV	GMOS-N
gmos-n_bpm_EEV_22_3amp_v1.fits	BPM     (2 2)	16	(2304, 1024)	EEV	GMOS-N
gmos-n_bpm_HAM_11_12amp_v1.fits	BPM     (1 1)	16	(4224, 512)	Hamamatsu-N	GMOS-N
gmos-n_bpm_HAM_11_12amp_v2.fits	BPM     (1 1)	16	(4224, 544)	Hamamatsu-N	GMOS-N
gmos-n_bpm_HAM_11_12amp_v3.fits	BPM     (1 1)	16	(4224, 512)	Hamamatsu-N	GMOS-N
gmos-n_bpm_HAM_12_12amp_v2.fits	BPM     (1 2)	16	(2112, 544)	Hamamatsu-N	GMOS-N
gmos-n_bpm_HAM_14_12amp_v2.fits	BPM     (1 4)	16	(1056, 544)	Hamamatsu-N	GMOS-N
gmos-n_bpm_HAM_21_12amp_v2.fits	BPM     (2 1)	16	(4224, 288)	Hamamatsu-N	GMOS-N
gmos-n_bpm_HAM_22_12amp_v1.fits	BPM     (2 2)	16	(2112, 256)	Hamamatsu-N	GMOS-N
gmos-n_bpm_HAM_22_12amp_v2.fits	BPM     (2 2)	16	(2112, 288)	Hamamatsu-N	GMOS-N
gmos-n_bpm_HAM_22_12amp_v3.fits	BPM     (2 2)	16	(2112, 256)	Hamamatsu-N	GMOS-N
gmos-n_bpm_HAM_24_12amp_v2.fits	BPM     (2 4)	16	(1056, 288)	Hamamatsu-N	GMOS-N
gmos-n_bpm_HAM_41_12amp_v2.fits	BPM     (4 1)	16	(4224, 160)	Hamamatsu-N	GMOS-N
gmos-n_bpm_HAM_42_12amp_v2.fits	BPM     (4 2)	16	(2112, 160)	Hamamatsu-N	GMOS-N
gmos-n_bpm_HAM_44_12amp_v1.fits	BPM     (4 4)	16	(1056, 128)	Hamamatsu-N	GMOS-N
gmos-n_bpm_HAM_44_12amp_v2.fits	BPM     (4 4)	16	(1056, 160)	Hamamatsu-N	GMOS-N
gmos-n_bpm_HAM_44_12amp_v3.fits	BPM     (4 4)	16	(1056, 128)	Hamamatsu-N	GMOS-N
gmos-n_bpm_e2v_11_6amp_v1.fits	BPM     (1 1)	16	(4608, 1024)	e2vDD	GMOS-N
gmos-n_bpm_e2v_11_6amp_v2.fits	BPM     (1 1)	16	(4608, 1024)	e2vDD	GMOS-N
gmos-n_bpm_e2v_22_6amp_v1.fits	BPM     (2 2)	16	(2304, 512)	e2vDD	GMOS-N
gmos-n_bpm_e2v_22_6amp_v2.fits	BPM     (2 2)	16	(2304, 512)	e2vDD	GMOS-N
gmos-s_bpm_EEV_11_3amp_v1.fits	BPM     (1 1)	16	(4608, 2048)	EEV	GMOS-S
gmos-s_bpm_EEV_11_3amp_v2.fits	BPM     (1 1)	16	(4608, 2048)	EEV	GMOS-S
gmos-s_bpm_EEV_12_3amp_v2.fits	bad_column_mask     (1 2)	16	(2304, 2048)	EEV	GMOS-S
gmos-s_bpm_EEV_12_3amp_v3.fits	BPM     (1 2)	16	(2304, 2048)	EEV	GMOS-S
gmos-s_bpm_EEV_14_3amp_v2.fits	bad_column_mask     (1 4)	16	(1152, 2048)	EEV	GMOS-S
gmos-s_bpm_EEV_14_3amp_v3.fits	BPM     (1 4)	16	(1152, 2048)	EEV	GMOS-S
gmos-s_bpm_EEV_22_3amp_v1.fits	BPM     (2 2)	16	(2304, 1024)	EEV	GMOS-S
gmos-s_bpm_EEV_22_3amp_v2.fits	BPM     (2 2)	16	(2304, 1024)	EEV	GMOS-S
gmos-s_bpm_EEV_24_3amp_v2.fits	bad_column_mask     (2 4)	16	(1152, 1024)	EEV	GMOS-S
gmos-s_bpm_EEV_24_3amp_v3.fits	BPM     (2 4)	16	(1152, 1024)	EEV	GMOS-S
gmos-s_bpm_EEV_41_3amp_v2.fits	bad_column_mask     (4 1)	16	(4608, 512)	EEV	GMOS-S
gmos-s_bpm_EEV_41_3amp_v3.fits	BPM     (4 1)	16	(4608, 512)	EEV	GMOS-S
gmos-s_bpm_EEV_42_3amp_v2.fits	bad_column_mask     (4 2)	16	(2304, 512)	EEV	GMOS-S
gmos-s_bpm_EEV_42_3amp_v3.fits	BPM     (4 2)	16	(2304, 512)	EEV	GMOS-S
gmos-s_bpm_EEV_44_3amp_v2.fits	bad_column_mask     (4 4)	16	(1152, 512)	EEV	GMOS-S
gmos-s_bpm_EEV_44_3amp_v3.fits	BPM     (4 4)	16	(1152, 512)	EEV	GMOS-S
gmos-s_bpm_HAM_11_12amp_v1.fits	GMOS-S BPM 1x1 binning     (1 1)	16	(4224, 512)	Hamamatsu-S	GMOS-S
gmos-s_bpm_HAM_11_12amp_v2.fits	GMOS-S BPM 1x1 binning     (1 1)	16	(4224, 512)	Hamamatsu-S	GMOS-S
gmos-s_bpm_HAM_11_12amp_v3.fits	BPM     (1 1)	16	(4224, 512)	Hamamatsu-S	GMOS-S
gmos-s_bpm_HAM_14_12amp_v2.fits	bad_column_mask     (1 4)	16	(1056, 512)	Hamamatsu-S	GMOS-S
gmos-s_bpm_HAM_14_12amp_v3.fits	BPM     (1 4)	16	(1056, 512)	Hamamatsu-S	GMOS-S
gmos-s_bpm_HAM_22_12amp_v1.fits	GMOS-S BPM 2x2 binning     (2 2)	16	(2112, 256)	Hamamatsu-S	GMOS-S
gmos-s_bpm_HAM_22_12amp_v2.fits	GMOS-S BPM 2x2 binning     (2 2)	16	(2112, 256)	Hamamatsu-S	GMOS-S
gmos-s_bpm_HAM_22_12amp_v3.fits	BPM     (2 2)	16	(2112, 256)	Hamamatsu-S	GMOS-S
gmos-s_bpm_HAM_24_12amp_v2.fits	bad_column_mask     (2 4)	16	(1056, 256)	Hamamatsu-S	GMOS-S
gmos-s_bpm_HAM_24_12amp_v3.fits	BPM     (2 4)	16	(1056, 256)	Hamamatsu-S	GMOS-S
gmos-s_bpm_HAM_41_12amp_v2.fits	bad_column_mask     (4 1)	16	(4224, 128)	Hamamatsu-S	GMOS-S
gmos-s_bpm_HAM_41_12amp_v3.fits	BPM     (4 1)	16	(4224, 128)	Hamamatsu-S	GMOS-S
gmos-s_bpm_HAM_42_12amp_v2.fits	bad_column_mask     (4 2)	16	(2112, 128)	Hamamatsu-S	GMOS-S
gmos-s_bpm_HAM_42_12amp_v3.fits	BPM     (4 2)	16	(2112, 128)	Hamamatsu-S	GMOS-S
gmos-s_bpm_HAM_44_12amp_v1.fits	GMOS-S BPM 4x4 binning     (4 4)	16	(1056, 128)	Hamamatsu-S	GMOS-S
gmos-s_bpm_HAM_44_12amp_v2.fits	GMOS-S BPM 4x4 binning     (4 4)	16	(1056, 128)	Hamamatsu-S	GMOS-S
gmos-s_bpm_HAM_44_12amp_v3.fits	BPM     (4 4)	16	(1056, 128)	Hamamatsu-S	GMOS-S
======================
"""

import os
import datetime

import astrodata
import gemini_instruments

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

    if delta:
        ad.write(filename="interim_"+ad.filename)
    return

def dragons_bpm(ad):
    """
    Add header things to a BPM to convert it into a DRAGONS compatible
    BPM. Sets header keywords,

    OBJECT
    BPMASK

    Writes out a file with the DRAGONS naming convention.

    """
    namps = None
    nbuffer = "{}_{}_{}_{}_{}_{}.fits"

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
        str(ad.detector_x_bin())+str(ad.detector_y_bin()),namps,"v3")
    try:
        ad.write(filename=new_bpm_name)
    except OSError as err:
        print("Warning: filename is over subscribed. Higher version takes precedence.")
        ad.write(filename=new_bpm_name, overwrite=True)

    return new_bpm_name


def tabl(ffiles):
    dirheader = "    \n\n  DIRECTORY {}/\n "
    header = "\tFile\t\t\tOBJ   BININNG  BITPIX    Shape 0         Det Name"
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
