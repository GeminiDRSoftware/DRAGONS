===================================
DRAGONS version of IGRINS2 pipeline
===================================

.. container:: alert alert-success

   DRAGONS version is in very early stage of development. As of
   2024-05-01, this is a naive (and partial) translation of original
   IGRINS plp pipeline. It only support H band spectra, and stellar
   sources taken with ABBA nodding.

.. container:: alert alert-success

   Plesae understand that the primary purpose of this demo is receive
   (early) feedback on high-level workflow. But any comment will be
   welcomed.

Install
=======

If you have not, please consult “Preparation for dragons version.ipynb”
and install IGRINSDR.

Set up alias
============

To reduce number of typing, we make aliases for dataselect and reduce
that will load ingrisdr packages.

On the shell, you may do something like

.. code:: sh

   > alias dataselect_ig="dataselect --adpkg=igrins_instruments"
   > alias reduce_ig="reduce --drpkg=igrinsdr --adpkg=igrins_instruments"

Below we will us ipython magic commands for aliasing.

.. container:: cell

   .. code:: python

      alias dataselect_ig dataselect --adpkg=igrins_instruments

.. container:: cell

   .. code:: python

      alias reduce_ig reduce --drpkg=igrinsdr --adpkg=igrins_instruments

.. container:: cell

   .. code:: python

      %matplotlib widget

.. container:: cell

   .. code:: python

      import astrodata
      import numpy as np
      import matplotlib.pyplot as plt

      import astrodata

Download data
=============

We will use engineering data of “gn-2024a-eng-142” which is available
from Gemini archive. Download data from 2024-04-29.

.. container:: alert alert-info

   If you have followed the plp version of notebook, you can skip this
   part as you already have data downloaded.

The MEF files from the archive need to be unbundled to H and K bands
files. For now we use a custom python script, which will be installed
when IGRINSDR gets installed. Assuming that files from gemini archive is
extracted in the directory “mef_dir” (the files need to be unzipped if
zipped). ‘mef_extract’ script will be installed together with IGRINSDR.
This is identical to “mef_extract.py” included in plp version.

.. code:: bash

   > mef_extract mef_dir 20240429 indata/20240429

.. container:: cell

   .. code:: python

      !mef_extract 20240429 --mefdir 20240429_demo

   .. container:: cell-output cell-output-stdout

      ::

         N20240429S0120.fits
         N20240429S0121.fits
         N20240429S0122.fits
         N20240429S0123.fits
         N20240429S0124.fits
         N20240429S0125.fits
         N20240429S0126.fits
         N20240429S0127.fits
         N20240429S0128.fits
         N20240429S0188.fits
         N20240429S0189.fits
         N20240429S0190.fits
         N20240429S0191.fits
         N20240429S0192.fits
         N20240429S0193.fits
         N20240429S0194.fits
         N20240429S0195.fits
         N20240429S0196.fits
         N20240429S0197.fits
         N20240429S0204.fits
         N20240429S0365.fits
         N20240429S0366.fits
         N20240429S0367.fits
         N20240429S0368.fits
         N20240429S0369.fits
         N20240429S0370.fits
         N20240429S0371.fits
         N20240429S0372.fits
         N20240429S0373.fits
         N20240429S0374.fits
         N20240429S0375.fits
         N20240429S0376.fits
         N20240429S0377.fits
         N20240429S0378.fits
         N20240429S0379.fits
         N20240429S0380.fits
         N20240429S0381.fits
         N20240429S0382.fits
         N20240429S0383.fits
         N20240429S0384.fits

.. container:: cell

   .. code:: python

      !ls unbundled_20240429/

   .. container:: cell-output cell-output-stdout

      ::

         N20240429S0122_H.fits  N20240429S0195_H.fits  N20240429S0373_H.fits
         N20240429S0122_K.fits  N20240429S0195_K.fits  N20240429S0373_K.fits
         N20240429S0123_H.fits  N20240429S0196_H.fits  N20240429S0374_H.fits
         N20240429S0123_K.fits  N20240429S0196_K.fits  N20240429S0374_K.fits
         N20240429S0124_H.fits  N20240429S0197_H.fits  N20240429S0375_H.fits
         N20240429S0124_K.fits  N20240429S0197_K.fits  N20240429S0375_K.fits
         N20240429S0125_H.fits  N20240429S0204_H.fits  N20240429S0376_H.fits
         N20240429S0125_K.fits  N20240429S0204_K.fits  N20240429S0376_K.fits
         N20240429S0126_H.fits  N20240429S0365_H.fits  N20240429S0377_H.fits
         N20240429S0126_K.fits  N20240429S0365_K.fits  N20240429S0377_K.fits
         N20240429S0127_H.fits  N20240429S0366_H.fits  N20240429S0378_H.fits
         N20240429S0127_K.fits  N20240429S0366_K.fits  N20240429S0378_K.fits
         N20240429S0128_H.fits  N20240429S0367_H.fits  N20240429S0379_H.fits
         N20240429S0128_K.fits  N20240429S0367_K.fits  N20240429S0379_K.fits
         N20240429S0190_H.fits  N20240429S0368_H.fits  N20240429S0380_H.fits
         N20240429S0190_K.fits  N20240429S0368_K.fits  N20240429S0380_K.fits
         N20240429S0191_H.fits  N20240429S0369_H.fits  N20240429S0381_H.fits
         N20240429S0191_K.fits  N20240429S0369_K.fits  N20240429S0381_K.fits
         N20240429S0192_H.fits  N20240429S0370_H.fits  N20240429S0382_H.fits
         N20240429S0192_K.fits  N20240429S0370_K.fits  N20240429S0382_K.fits
         N20240429S0193_H.fits  N20240429S0371_H.fits  N20240429S0383_H.fits
         N20240429S0193_K.fits  N20240429S0371_K.fits  N20240429S0383_K.fits
         N20240429S0194_H.fits  N20240429S0372_H.fits  N20240429S0384_H.fits
         N20240429S0194_K.fits  N20240429S0372_K.fits  N20240429S0384_K.fits

.. container:: cell

   .. code:: python

      from igrinsdr_helper.igrinsdr_tree import get_ad_tree
      from pathlib import Path

      get_ad_tree(Path("./unbundled_20240429").glob("N*_H.fits"))

   .. container:: cell-output cell-output-display

      ::

         Tree(nodes=(Node(icon_style='success', name="'H IGRINS GEMINI SPECT UNPREPARED NORTH VERSION2'", nodes=(Node(i…

Running REDUCE
==============

1. FLAT
2. SKY
3. STANDARD
4. TARGET

FLAT
====

Then create a file lising the FALT image from the fixed fits files.

   dataselect_ig –tags FLAT,H indata/20240429/SDCH\*.fits -o
   list_of_flat_h.txt

For now, you need to make a badpixel file using the flat images. For
this, you will the reduce with a specific recipe name.

   reduce_ig @list_of_flat_h.txt -r makeProcessedBPM

Run reduce_ig with the name of the created badpixel file as a parameter.

   reduce_ig @list_of_flat_h.txt -p
   user_bpm=SDCH_20240429_0365_hotpixel.fits

.. container:: cell

   .. code:: python

      %dataselect_ig --tags FLAT,H indata/20240429/SDCH*.fits -o list_of_flat_h.txt

.. container:: cell

   .. code:: python

      !cat list_of_flat_h.txt

   .. container:: cell-output cell-output-stdout

      ::

         # Includes tags: ['FLAT', 'H']
         # Excludes tags: []
         # Descriptor expression: None
         indata/20240429/SDCH_20240429_0365.fits
         indata/20240429/SDCH_20240429_0366.fits
         indata/20240429/SDCH_20240429_0367.fits
         indata/20240429/SDCH_20240429_0368.fits
         indata/20240429/SDCH_20240429_0369.fits
         indata/20240429/SDCH_20240429_0370.fits
         indata/20240429/SDCH_20240429_0371.fits
         indata/20240429/SDCH_20240429_0372.fits
         indata/20240429/SDCH_20240429_0373.fits
         indata/20240429/SDCH_20240429_0374.fits
         indata/20240429/SDCH_20240429_0375.fits
         indata/20240429/SDCH_20240429_0376.fits
         indata/20240429/SDCH_20240429_0377.fits
         indata/20240429/SDCH_20240429_0378.fits
         indata/20240429/SDCH_20240429_0379.fits
         indata/20240429/SDCH_20240429_0380.fits
         indata/20240429/SDCH_20240429_0381.fits
         indata/20240429/SDCH_20240429_0382.fits
         indata/20240429/SDCH_20240429_0383.fits
         indata/20240429/SDCH_20240429_0384.fits

.. container:: cell

   .. code:: python

      %reduce_ig @list_of_flat_h.txt -r makeProcessedBPM

   .. container:: cell-output cell-output-stdout

      ::


                     --- reduce v3.2.2 ---

         Running on Python 3.10.13
         All submitted files appear valid:
         unbundled_20240429/N20240429S0365_H.fits ... unbundled_20240429/N20240429S0384_H.fits, 20 files submitted.
         ================================================================================
         RECIPE: makeProcessedBPM
         ================================================================================
            PRIMITIVE: prepare
            ------------------
               PRIMITIVE: validateData
               -----------------------
               .
               PRIMITIVE: standardizeStructure
               -------------------------------
               .
               PRIMITIVE: standardizeHeaders
               -----------------------------
                  PRIMITIVE: standardizeObservatoryHeaders
                  ----------------------------------------
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  .
                  PRIMITIVE: standardizeInstrumentHeaders
                  ---------------------------------------
                  .
               .
               PRIMITIVE: standardizeWCS
               -------------------------
               .
            .
            PRIMITIVE: addDQ
            ----------------
            No BPMs found for N20240429S0365_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0366_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0367_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0368_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0369_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0370_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0371_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0372_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0373_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0374_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0375_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0376_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0377_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0378_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0379_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0380_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0381_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0382_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0383_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0384_H_prepared.fits and none supplied by the user.
            
            .
            PRIMITIVE: readoutPatternCorrectFlatOff
            ---------------------------------------
               PRIMITIVE: selectFromInputs
               ---------------------------
               .
            .
            PRIMITIVE: readoutPatternCorrectFlatOn
            --------------------------------------
            .
            PRIMITIVE: selectFromInputs
            ---------------------------
            .
            PRIMITIVE: stackFrames
            ----------------------
            Combining 10 inputs with mean and sigclip rejection
            Combining images.
            
            .
            PRIMITIVE: selectFromInputs
            ---------------------------
            .
            PRIMITIVE: stackFrames
            ----------------------
            Combining 10 inputs with mean and sigclip rejection
            Combining images.
            
            .
            PRIMITIVE: makeIgrinsBPM
            ------------------------
            .
            PRIMITIVE: storeBPM
            -------------------
               PRIMITIVE: storeCalibration
               ---------------------------
               ~/.dragons/dragons.db: Storing calibrations/processed_bpm/N20240429S0365_H_badpixel.fits as processed_bpm
               .
            .
             Wrote N20240429S0365_H_badpixel.fits in output directory

         reduce completed successfully.

.. container:: cell

   .. code:: python

      %reduce_ig @list_of_flat_h.txt -p user_bpm=N20240429S0365_H_badpixel.fits

   .. container:: cell-output cell-output-stdout

      ::


                     --- reduce v3.2.2 ---

         Running on Python 3.10.13
         All submitted files appear valid:
         unbundled_20240429/N20240429S0365_H.fits ... unbundled_20240429/N20240429S0384_H.fits, 20 files submitted.
         ================================================================================
         RECIPE: makeProcessedFlat
         ================================================================================
            PRIMITIVE: prepare
            ------------------
               PRIMITIVE: validateData
               -----------------------
               .
               PRIMITIVE: standardizeStructure
               -------------------------------
               .
               PRIMITIVE: standardizeHeaders
               -----------------------------
                  PRIMITIVE: standardizeObservatoryHeaders
                  ----------------------------------------
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  .
                  PRIMITIVE: standardizeInstrumentHeaders
                  ---------------------------------------
                  .
               .
               PRIMITIVE: standardizeWCS
               -------------------------
               .
            .
            PRIMITIVE: readoutPatternCorrectFlatOff
            ---------------------------------------
               PRIMITIVE: selectFromInputs
               ---------------------------
               .
            .
            PRIMITIVE: addDQ
            ----------------
            Using N20240429S0365_H_badpixel.fits as user BPM
            Using N20240429S0365_H_badpixel.fits as user BPM
            Using N20240429S0365_H_badpixel.fits as user BPM
            Using N20240429S0365_H_badpixel.fits as user BPM
            Using N20240429S0365_H_badpixel.fits as user BPM
            Using N20240429S0365_H_badpixel.fits as user BPM
            Using N20240429S0365_H_badpixel.fits as user BPM
            Using N20240429S0365_H_badpixel.fits as user BPM
            Using N20240429S0365_H_badpixel.fits as user BPM
            Using N20240429S0365_H_badpixel.fits as user BPM
            Using N20240429S0365_H_badpixel.fits as user BPM
            Using N20240429S0365_H_badpixel.fits as user BPM
            Using N20240429S0365_H_badpixel.fits as user BPM
            Using N20240429S0365_H_badpixel.fits as user BPM
            Using N20240429S0365_H_badpixel.fits as user BPM
            Using N20240429S0365_H_badpixel.fits as user BPM
            Using N20240429S0365_H_badpixel.fits as user BPM
            Using N20240429S0365_H_badpixel.fits as user BPM
            Using N20240429S0365_H_badpixel.fits as user BPM
            Using N20240429S0365_H_badpixel.fits as user BPM
            .
            PRIMITIVE: addVAR
            -----------------
            Adding the read noise component and the Poisson noise component of the variance
            .
            PRIMITIVE: ADUToElectrons
            -------------------------
            Converting N20240429S0365_Hrp_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0366_Hrp_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0367_Hrp_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0368_Hrp_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0369_Hrp_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0370_Hrp_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0371_Hrp_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0372_Hrp_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0373_Hrp_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0374_Hrp_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0375_H_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0376_H_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0377_H_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0378_H_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0379_H_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0380_H_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0381_H_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0382_H_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0383_H_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0384_H_varAdded.fits from ADU to electrons by multiplying by the gain
            .
            PRIMITIVE: makeLampFlat
            -----------------------
               PRIMITIVE: selectFromInputs
               ---------------------------
               .
               PRIMITIVE: selectFromInputs
               ---------------------------
               .
               PRIMITIVE: showInputs
               ---------------------
               Inputs for lampOn
                 N20240429S0375_H_ADUToElectrons.fits
                 N20240429S0376_H_ADUToElectrons.fits
                 N20240429S0377_H_ADUToElectrons.fits
                 N20240429S0378_H_ADUToElectrons.fits
                 N20240429S0379_H_ADUToElectrons.fits
                 N20240429S0380_H_ADUToElectrons.fits
                 N20240429S0381_H_ADUToElectrons.fits
                 N20240429S0382_H_ADUToElectrons.fits
                 N20240429S0383_H_ADUToElectrons.fits
                 N20240429S0384_H_ADUToElectrons.fits
               .
               PRIMITIVE: showInputs
               ---------------------
               Inputs for lampOff
                 N20240429S0365_Hrp_ADUToElectrons.fits
                 N20240429S0366_Hrp_ADUToElectrons.fits
                 N20240429S0367_Hrp_ADUToElectrons.fits
                 N20240429S0368_Hrp_ADUToElectrons.fits
                 N20240429S0369_Hrp_ADUToElectrons.fits
                 N20240429S0370_Hrp_ADUToElectrons.fits
                 N20240429S0371_Hrp_ADUToElectrons.fits
                 N20240429S0372_Hrp_ADUToElectrons.fits
                 N20240429S0373_Hrp_ADUToElectrons.fits
                 N20240429S0374_Hrp_ADUToElectrons.fits
               .
               PRIMITIVE: stackFrames
               ----------------------
               Combining 10 inputs with mean and sigclip rejection
               Combining images.
               
               .
               PRIMITIVE: stackFrames
               ----------------------
               Combining 10 inputs with mean and sigclip rejection
               Combining images.
               
               .
            .
            PRIMITIVE: determineSlitEdges
            -----------------------------
            .
            PRIMITIVE: maskBeyondSlit
            -------------------------
            .
            PRIMITIVE: normalizeFlat
            ------------------------
         /home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/procedures/normalize_flat.py:121: RuntimeWarning: All-NaN slice encountered
           s = np.nanmedian(dn,
         /home/jjlee/miniconda3/envs/dragons/lib/python3.10/site-packages/numpy/core/_methods.py:206: RuntimeWarning: Degrees of freedom <= 0 for slice
           ret = _var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
         /home/jjlee/miniconda3/envs/dragons/lib/python3.10/site-packages/numpy/core/_methods.py:163: RuntimeWarning: invalid value encountered in divide
           arrmean = um.true_divide(arrmean, div, out=arrmean,
         /home/jjlee/miniconda3/envs/dragons/lib/python3.10/site-packages/numpy/core/_methods.py:198: RuntimeWarning: invalid value encountered in divide
           ret = ret.dtype.type(ret / rcount)
            .
            PRIMITIVE: thresholdFlatfield
            -----------------------------
            .
            PRIMITIVE: storeProcessedFlat
            -----------------------------
               PRIMITIVE: storeCalibration
               ---------------------------
               ~/.dragons/dragons.db: Storing calibrations/processed_flat/N20240429S0375_H_flat.fits as processed_flat
               .
            .
             Wrote N20240429S0375_H_flat.fits in output directory

         reduce completed successfully.

.. container:: cell

   .. code:: python

      ad_flat = astrodata.open("N20240429S0375_H_flat.fits")
      ad_flat.info()

   .. container:: cell-output cell-output-stdout

      ::

         Filename: N20240429S0375_H_flat.fits
         Tags:

         Pixels Extensions
         Index  Content                  Type              Dimensions     Format
         [ 0]   science                  NDAstroData       (2048, 2048)   float64
                   .variance             ADVarianceUncerta (2048, 2048)   float32
                   .mask                 ndarray           (2048, 2048)   uint16
                   .FLAT_ORIGINAL        ndarray           (2048, 2048)   float32
                   .SLITEDGE             Table             (54, 6)        n/a

         Other Extensions
                        Type        Dimensions
         .FLATNORM      Table       (3, 3)
         .HISTORY       Table       (8, 4)
         .PROVENANCE    Table       (21, 4)

.. container:: cell

   .. code:: python

      plt.figure()
      plt.imshow(ad_flat[0].FLAT_ORIGINAL, origin="lower")

   .. container:: cell-output cell-output-display

      ::

         <matplotlib.image.AxesImage at 0x71f9c895ed40>

   .. container:: cell-output cell-output-display

      .. image:: IGRINSDR_users_manual_files/figure-rst/cell-14-output-2.png

.. container:: cell

   .. code:: python

      plt.figure()
      plt.imshow(np.ma.array(ad_flat[0].data, mask=ad_flat[0].mask).filled(np.nan), vmin=0.8, vmax=1.2, origin="lower")

   .. container:: cell-output cell-output-display

      ::

         <matplotlib.image.AxesImage at 0x71f9c8752a40>

   .. container:: cell-output cell-output-display

      .. image:: IGRINSDR_users_manual_files/figure-rst/cell-15-output-2.png

SKY
===

(this recipe is a combined version of register-sky and wvlsol-sky from
the PLP)

Create a file containg sky frames.

   dataselect_ig –tags SKY,H indata/20240429/SDC\*.fits -o
   list_of_sky_h.txt

.. container:: alert alert-info

   CALDB support is not properly integrated yet. You need to explicitly
   speicy calibrations file with the “–user_cal” options.

We will run reduce, but we need to explicitly set the calibration file.

   reduce_ig @list_of_sky_h.txt –user_cal
   processed_flat:SDCH_20240429_0375_flat.fits

.. container:: cell

   .. code:: python

      %dataselect_ig --tags SKY,H indata/20240429/SDCH*.fits -o list_of_sky_h.txt

.. container:: cell

   .. code:: python

      !cat list_of_sky_h.txt

   .. container:: cell-output cell-output-stdout

      ::

         # Includes tags: ['SKY', 'H']
         # Excludes tags: []
         # Descriptor expression: None
         indata/20240429/SDCH_20240429_0204.fits

.. container:: cell

   .. code:: python

      %reduce_ig @list_of_sky_h.txt --user_cal processed_flat:N20240429S0375_H_flat.fits

   .. container:: cell-output cell-output-stdout

      ::


                     --- reduce v3.2.2 ---

         Running on Python 3.10.13
         All submitted files appear valid:
         unbundled_20240429/N20240429S0204_H.fits
         Manually assigned N20240429S0375_H_flat.fits as processed_flat
         ================================================================================
         RECIPE: makeProcessedArc
         ================================================================================
            PRIMITIVE: prepare
            ------------------
               PRIMITIVE: validateData
               -----------------------
               .
               PRIMITIVE: standardizeStructure
               -------------------------------
               .
               PRIMITIVE: standardizeHeaders
               -----------------------------
                  PRIMITIVE: standardizeObservatoryHeaders
                  ----------------------------------------
                  Updating keywords that are common to all Gemini data
                  .
                  PRIMITIVE: standardizeInstrumentHeaders
                  ---------------------------------------
                  .
               .
               PRIMITIVE: standardizeWCS
               -------------------------
               .
            .
            PRIMITIVE: addDQ
            ----------------
            No BPMs found for N20240429S0204_H_prepared.fits and none supplied by the user.
            
            .
            PRIMITIVE: addVAR
            -----------------
            Adding the read noise component of the variance
            .
            PRIMITIVE: ADUToElectrons
            -------------------------
            Converting N20240429S0204_H_varAdded.fits from ADU to electrons by multiplying by the gain
            .
            PRIMITIVE: stackFrames
            ----------------------
            No stacking will be performed, since at least two input AstroData objects are required for stackFrames
            .
            PRIMITIVE: extractSimpleSpec
            ----------------------------
            .
            PRIMITIVE: identifyOrders
            -------------------------
            .
            PRIMITIVE: identifyLines
            ------------------------
            .
            PRIMITIVE: getInitialWvlsol
            ---------------------------
            .
            PRIMITIVE: extractSpectraMulti
            ------------------------------
            .
            PRIMITIVE: identifyMultiline
            ----------------------------
            .
            PRIMITIVE: volumeFit
            --------------------
         /home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/procedures/process_wvlsol_volume_fit.py:66: FutureWarning: The provided callable <function std at 0x7f41ae7cb490> is currently using SeriesGroupBy.std. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "std" instead.
           ss0_std = ss0.transform(np.std)
            .
            PRIMITIVE: makeSpectralMaps
            ---------------------------
            .
            PRIMITIVE: attachWatTable
            -------------------------
            .
            PRIMITIVE: storeProcessedArc
            ----------------------------
               PRIMITIVE: storeCalibration
               ---------------------------
               ~/.dragons/dragons.db: Storing calibrations/processed_arc/N20240429S0204_H_arc.fits as processed_arc
               .
            .
             Wrote N20240429S0204_H_arc.fits in output directory

         reduce completed successfully.

.. container:: cell

   .. code:: python

      ad_sky = astrodata.open("N20240429S0204_H_arc.fits")
      ad_sky.info()

   .. container:: cell-output cell-output-stdout

      ::

         Filename: N20240429S0204_H_arc.fits
         Tags:

         Pixels Extensions
         Index  Content                  Type              Dimensions     Format
         [ 0]   science                  NDAstroData       (2048, 2048)   float32
                   .variance             ADVarianceUncerta (2048, 2048)   float32
                   .mask                 ndarray           (2048, 2048)   uint16
                   .LINEFIT              Table             (755, 8)       n/a
                   .LINEID               Table             (1311, 4)      n/a
                   .ORDERMAP             ndarray           (2048, 2048)   int32
                   .SLITEDGE             Table             (54, 6)        n/a
                   .SLITOFFSETMAP        ndarray           (2048, 2048)   float64
                   .SLITPOSMAP           ndarray           (2048, 2048)   float64
                   .SPEC1D               Table             (27, 4)        n/a
                   .SPEC1D_MULTI         Table             (27, 3)        n/a
                   .VOLUMEFIT_COEFFS     Table             (24, 4)        n/a
                   .WAT_HEADER           Table             (91, 1)        n/a
                   .WVLFIT_RESULTS       Table             (5, 3)         n/a
                   .WVLSOL               Table             (27, 2)        n/a
                   .WVLSOL0              Table             (27, 2)        n/a

         Other Extensions
                        Type        Dimensions
         .HISTORY       Table       (7, 4)
         .PROVENANCE    Table       (1, 4)

.. container:: cell

   .. code:: python

      plt.figure()
      plt.imshow(ad_sky[0].ORDERMAP, origin="lower")

   .. container:: cell-output cell-output-display

      ::

         <matplotlib.image.AxesImage at 0x71f97e76c9a0>

   .. container:: cell-output cell-output-display

      .. image:: IGRINSDR_users_manual_files/figure-rst/cell-20-output-2.png

.. container:: cell

   .. code:: python

      plt.figure()
      plt.imshow(ad_sky[0].SLITPOSMAP, origin="lower")

   .. container:: cell-output cell-output-display

      ::

         <matplotlib.image.AxesImage at 0x71f9c8684df0>

   .. container:: cell-output cell-output-display

      .. image:: IGRINSDR_users_manual_files/figure-rst/cell-21-output-2.png

.. container:: cell

   .. code:: python

      ad_sky[0].WVLSOL

   .. container:: cell-output cell-output-display

      orders
      wavelengths
      int64
      float64[2048]
      98
      1.8111933301142387 .. 1.8360399695657343
      99
      1.7933534545680807 .. 1.8179632688639047
      100
      1.7758753841772261 .. 1.800252099490029
      101
      1.7587484158815256 .. 1.7828956032269987
      102
      1.7419622663486773 .. 1.7658833476701403
      103
      1.7255070515990916 .. 1.74920530555671
      ...
      ...
      118
      1.5126951517015044 .. 1.533361864468183
      119
      1.5004531601245925 .. 1.5209338227376323
      120
      1.488420109176072 .. 1.5087162318068306
      121
      1.4765908549262368 .. 1.4967038731770737
      122
      1.4649604220988142 .. 1.484891699447973
      123
      1.4535239972151333 .. 1.4732748273622442
      124
      1.442276922070029 .. 1.461848531187029

A0V
===

Let’s do telluric standar star. We will select images from
observation_id of ‘GN-2024A-ENG-142-261’.

   dataselect_ig –tags STANDARD,H indata/20240425/SDC\*.fits -o
   list_of_std_h.txt –expr “observation_id==‘GN-2024A-ENG-142-261’”

Again, we need to explicitly specify calibration files.

   reduce_ig @list_of_std_h.txt –user_cal
   processed_flat:SDCH_20240429_0375_flat.fits
   processed_arc:SDCH_20240429_0204_arc.fits

.. container:: cell

   .. code:: python

      %dataselect_ig --tags STANDARD,H indata/20240429/SDC*.fits -o list_of_std_h.txt --expr "observation_id=='GN-2024A-ENG-142-261'"

.. container:: cell

   .. code:: python

      !cat list_of_std_h.txt

   .. container:: cell-output cell-output-stdout

      ::

         # Includes tags: ['STANDARD', 'H']
         # Excludes tags: []
         # Descriptor expression: observation_id=='GN-2024A-ENG-142-203'
         unbundled_20240429/N20240429S0125_H.fits
         unbundled_20240429/N20240429S0126_H.fits
         unbundled_20240429/N20240429S0127_H.fits
         unbundled_20240429/N20240429S0128_H.fits

.. container:: cell

   .. code:: python

      %reduce_ig @list_of_std_h.txt --user_cal processed_flat:N20240429S0375_H_flat.fits processed_arc:N20240429S0204_H_arc.fits

   .. container:: cell-output cell-output-stdout

      ::


                     --- reduce v3.2.2 ---

         Running on Python 3.10.13
         All submitted files appear valid:
         unbundled_20240429/N20240429S0125_H.fits ... unbundled_20240429/N20240429S0128_H.fits, 4 files submitted.
         Manually assigned N20240429S0375_H_flat.fits as processed_flat
         Manually assigned N20240429S0204_H_arc.fits as processed_arc
         ================================================================================
         RECIPE: makeStd
         ================================================================================
            PRIMITIVE: checkCALDB
            ---------------------
            .
            PRIMITIVE: prepare
            ------------------
               PRIMITIVE: validateData
               -----------------------
               .
               PRIMITIVE: standardizeStructure
               -------------------------------
               .
               PRIMITIVE: standardizeHeaders
               -----------------------------
                  PRIMITIVE: standardizeObservatoryHeaders
                  ----------------------------------------
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  Updating keywords that are common to all Gemini data
                  .
                  PRIMITIVE: standardizeInstrumentHeaders
                  ---------------------------------------
                  .
               .
               PRIMITIVE: standardizeWCS
               -------------------------
               .
            .
            PRIMITIVE: addDQ
            ----------------
            No BPMs found for N20240429S0125_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0126_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0127_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0128_H_prepared.fits and none supplied by the user.
            
            .
            PRIMITIVE: addVAR
            -----------------
            Adding the read noise component and the Poisson noise component of the variance
            .
            PRIMITIVE: ADUToElectrons
            -------------------------
            Converting N20240429S0125_H_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0126_H_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0127_H_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0128_H_varAdded.fits from ADU to electrons by multiplying by the gain
            .
            PRIMITIVE: makeAB
            -----------------
               PRIMITIVE: stackFrames
               ----------------------
               Combining 2 inputs with mean and sigclip rejection
               Combining images.
               
               .
               PRIMITIVE: stackFrames
               ----------------------
               Combining 2 inputs with mean and sigclip rejection
               Combining images.
               
               .
         /home/jjlee/miniconda3/envs/dragons/lib/python3.10/site-packages/astrodata/nddata.py:32: RuntimeWarning: Negative variance values found. Setting to zero.
           warnings.warn("Negative variance values found. Setting to zero.",
            .
            PRIMITIVE: estimateSlitProfile
            ------------------------------
            .
            PRIMITIVE: extractStellarSpec
            -----------------------------
            .
            PRIMITIVE: saveTwodspec
            -----------------------
            .
            PRIMITIVE: saveDebugImage
            -------------------------
            .
             Wrote N20240429S0125_H_spec1d.fits in output directory

         reduce completed successfully.

.. container:: cell

   .. code:: python

      ad_std = astrodata.open("N20240429S0125_H_spec1d.fits")
      ad_std.info()

   .. container:: cell-output cell-output-stdout

      ::

         Filename: N20240429S0125_H_spec1d.fits
         Tags:

         Pixels Extensions
         Index  Content                  Type              Dimensions     Format
         [ 0]   science                  NDAstroData       (27, 2048)     float64
                   .variance             ADVarianceUncerta (27, 2048)     float64
                   .SN_PER_RESEL         ndarray           (27, 2048)     float64
                   .WAVELENGTHS          ndarray           (27, 2048)     float64

.. container:: cell

   .. code:: python

      plt.figure()
      plt.imshow(ad_std[0].data, origin="lower", aspect="auto", interpolation="none")
      plt.gca().set(xlabel="wavelength axis", ylabel="order axis")

   .. container:: cell-output cell-output-display

      ::

         [Text(0.5, 0, 'wavelength axis'), Text(0, 0.5, 'order axis')]

   .. container:: cell-output cell-output-display

      .. image:: IGRINSDR_users_manual_files/figure-rst/cell-27-output-2.png

.. container:: cell

   .. code:: python

      fig, ax = plt.subplots(figsize=(12, 4))
      for w, s in zip(ad_std[0].WAVELENGTHS, ad_std[0].data):
          ax.plot(w[6:-6], s[6:-6])

   .. container:: cell-output cell-output-display

      .. image:: IGRINSDR_users_manual_files/figure-rst/cell-28-output-1.png

.. container:: cell

   .. code:: python

      fig

   .. container:: cell-output cell-output-display

      .. image:: IGRINSDR_users_manual_files/figure-rst/cell-29-output-1.png

.. container:: cell

   .. code:: python

      ax.set_xlim(1.7, 1.75)
      ax.set_ylim(-4000, 24000)
      fig

   .. container:: cell-output cell-output-display

      .. image:: IGRINSDR_users_manual_files/figure-rst/cell-30-output-1.png

.. container:: cell

   .. code:: python

      ad_std_2dspec = astrodata.open("N20240429S0125_H_spec2d.fits")
      ad_std_2dspec.info()

   .. container:: cell-output cell-output-stdout

      ::

         Filename: N20240429S0125_H_spec2d.fits
         Tags:

         Pixels Extensions
         Index  Content                  Type              Dimensions     Format
         [ 0]   science                  NDAstroData       (27, 63, 2048) float32
                   .variance             ADVarianceUncerta (27, 63, 2048) float64
                   .WAVELENGTHS          ndarray           (27, 2048)     float64

.. container:: cell

   .. code:: python

      from mpl_toolkits.axes_grid1 import Grid
      fig = plt.figure()
      grid = Grid(fig, 111, (10, 3), direction="column")
      for ax, im in zip(grid, ad_std_2dspec[0].data):
          ax.imshow(im, origin="lower", aspect="auto", cmap="coolwarm")

   .. container:: cell-output cell-output-display

      .. image:: IGRINSDR_users_manual_files/figure-rst/cell-32-output-1.png

.. container:: cell

   .. code:: python

      ad_std_debug = astrodata.open("N20240429S0125_H_spec_debug.fits")
      ad_std_debug.info()

   .. container:: cell-output cell-output-stdout

      ::

         Filename: N20240429S0125_H_spec_debug.fits
         Tags:

         Pixels Extensions
         Index  Content                  Type              Dimensions     Format
         [ 0]   science                  NDAstroData       (2048, 2048)   float64
                   .variance             ADVarianceUncerta (2048, 2048)   float64
                   .mask                 ndarray           (2048, 2048)   uint16
                   .SLITPROFILE          Table             (5, 3)         n/a
                   .SLITPROFILE_MAP      ndarray           (2048, 2048)   float64
                   .SPEC1D               Table             (27, 5)        n/a
                   .WVLCOR               Table             (4, 2)         n/a

SCIENCE
=======

For science target, select 900078.

   dataselect_ig –tags SCIENCE,H indata/20240429/SDC\*.fits -o
   list_of_900078_h.txt –expr “object==‘900078’”

Run reduce.

   reduce_ig @list_of_900078_h.txt –user_cal
   processed_flat:SDCH_20240429_0375_flat.fits
   processed_arc:SDCH_20240429_0204_arc.fits

.. container:: cell

   .. code:: python

      %dataselect_ig --tags SCIENCE,H indata/20240429/SDCH*.fits -o list_of_900078_h.txt --expr "object=='T Coronae Borealis'"

.. container:: cell

   .. code:: python

      !cat list_of_900078_h.txt

   .. container:: cell-output cell-output-stdout

      ::

         # Includes tags: ['SCIENCE', 'H']
         # Excludes tags: []
         # Descriptor expression: object=='T Coronae Borealis'
         indata/20240429/SDCH_20240429_0213.fits
         indata/20240429/SDCH_20240429_0214.fits
         indata/20240429/SDCH_20240429_0215.fits
         indata/20240429/SDCH_20240429_0216.fits
         indata/20240429/SDCH_20240429_0217.fits
         indata/20240429/SDCH_20240429_0218.fits
         indata/20240429/SDCH_20240429_0219.fits
         indata/20240429/SDCH_20240429_0220.fits

.. container:: cell

   .. code:: python

      %reduce_ig @list_of_900078_h.txt --user_cal processed_flat:N20240429S0375_H_flat.fits processed_arc:N20240429S0204_H_arc.fits

   .. container:: cell-output cell-output-stdout

      ::


                     --- reduce v3.2.2 ---

         Running on Python 3.10.13
         All submitted files appear valid:
         indata/20240429/SDCH_20240429_0213.fits ... indata/20240429/SDCH_20240429_0220.fits, 8 files submitted.
         Manually assigned N20240429S0375_H_flat.fits as processed_flat
         Manually assigned N20240429S0204_H_arc.fits as processed_arc
         ================================================================================
         RECIPE: fixHeader
         ================================================================================
            PRIMITIVE: fixHeader
            --------------------
            .
            PRIMITIVE: writeOutputs
            -----------------------
            Writing to file SDCH_20240429_0213_fixed.fits
            Writing to file SDCH_20240429_0214_fixed.fits
            Writing to file SDCH_20240429_0215_fixed.fits
            Writing to file SDCH_20240429_0216_fixed.fits
            Writing to file SDCH_20240429_0217_fixed.fits
            Writing to file SDCH_20240429_0218_fixed.fits
            Writing to file SDCH_20240429_0219_fixed.fits
            Writing to file SDCH_20240429_0220_fixed.fits
            .
             Wrote SDCH_20240429_0213_fixed.fits in output directory
             Wrote SDCH_20240429_0214_fixed.fits in output directory
             Wrote SDCH_20240429_0215_fixed.fits in output directory
             Wrote SDCH_20240429_0216_fixed.fits in output directory
             Wrote SDCH_20240429_0217_fixed.fits in output directory
             Wrote SDCH_20240429_0218_fixed.fits in output directory
             Wrote SDCH_20240429_0219_fixed.fits in output directory
             Wrote SDCH_20240429_0220_fixed.fits in output directory

         reduce completed successfully.

.. container:: cell

   .. code:: python

      ad_tgt = astrodata.open("SDCH_20240429_0213_spec1d.fits")
      ad_tgt.info()

   .. container:: cell-output cell-output-stdout

      ::

         Filename: SDCH_20240429_0213_spec1d.fits
         Tags: GEMINI H IGRINS NORTH PREPARED SCIENCE SIDEREAL SPECT VERSION2

         Pixels Extensions
         Index  Content                  Type              Dimensions     Format
         [ 0]   science                  NDAstroData       (27, 2048)     float64
                   .variance             ADVarianceUncerta (27, 2048)     float64
                   .SN_PER_RESEL         ndarray           (27, 2048)     float64
                   .WAVELENGTHS          ndarray           (27, 2048)     float64

.. container:: cell

   .. code:: python

      plt.imshow(ad_tgt[0].data, origin="lower", aspect="auto", interpolation="none")
      plt.gca().set(xlabel="wavelength axis", ylabel="order axis")

   .. container:: cell-output cell-output-display

      ::

         [Text(0.5, 0, 'wavelength axis'), Text(0, 0.5, 'order axis')]

   .. container:: cell-output cell-output-display

      .. image:: IGRINSDR_users_manual_files/figure-rst/cell-38-output-2.png

.. container:: cell

   .. code:: python

      fig, ax = plt.subplots(figsize=(12, 4))
      for w, s, t in zip(ad_std[0].WAVELENGTHS, ad_std[0].data, ad_tgt[0].data):
          ax.plot(w, t/s)
      ax.set_ylim(0., 0.7)

   .. container:: cell-output cell-output-display

      ::

         (0.0, 0.7)

   .. container:: cell-output cell-output-display

      .. image:: IGRINSDR_users_manual_files/figure-rst/cell-39-output-2.png

.. container:: cell

   .. code:: python

      ax.set_xlim(1.72, 1.75)
      ax.set_ylim(0.2, 0.8)
      fig

   .. container:: cell-output cell-output-display

      .. image:: IGRINSDR_users_manual_files/figure-rst/cell-40-output-1.png
