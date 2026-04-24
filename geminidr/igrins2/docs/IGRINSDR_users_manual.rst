==============================================
DRAGONS-compatible version of IGRINS2 pipeline
==============================================

.. container:: alert alert-success

   DRAGONS-compatible version is in very early stage of development.
   This is largely a translation of original IGRINS plp pipeline to the
   DRAGONS framework. For now, it supports H and K band spectra, and the
   results should be comparable to that of the plp version of the
   pipeline.

.. container:: alert alert-success

   At this stage of development, plesae understand that the primary
   purpose of this document is to receive (early) feedback on high-level
   workflow. But any comment will be welcomed.

Install
=======

Please consult :doc:``INSTALL`` to install IGRINSDR.

Setting up
==========

The original version of this document is a jupyter notebook, mixed with
shell comannds and python code. It may be better to look at the original
notebook file.

To reduce number of typing, we make aliases for ``dataselect`` and
``reduce`` so that they load ingrisdr packages automtically.

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

Below are initial import statements for the pyton codes.

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

The MEF files from the archive need to be unbundled to H and K bands
files. Assuming that files from gemini archive is extracted in the
directory “mef_20240429” (the files need to be unzipped if zipped).
‘reduce’ can be used to unbundle these files.

.. code:: bash

   > dataselect_ig --tags BUNDLE mef_20240429/N*.fits -o list_of_bundles.txt
   > reduce_ig @list_of_bundles.txt

For the commands below, we will temporarily cd to
``unbundled_20240429``. The unbundled files will be saved in the working
directory.

.. container:: cell

   .. code:: python

      !mkdir unbundled_20240429
      %cd unbundled_20240429

   .. container:: cell-output cell-output-stdout

      ::

         /home/jjlee/git_personal/IGRINSDR/test_i2/unbundled_20240429

.. container:: cell

   .. code:: python

      %dataselect_ig --tags BUNDLE ../mef_20240429/N*.fits -o list_of_bundles.txt

.. container:: cell

   .. code:: python

      !cat list_of_bundles.txt

   .. container:: cell-output cell-output-stdout

      ::

         # Includes tags: ['BUNDLE']
         # Excludes tags: []
         # Descriptor expression: None
         ../mef_20240429/N20240429S0120.fits
         ../mef_20240429/N20240429S0121.fits
         ../mef_20240429/N20240429S0122.fits
         ../mef_20240429/N20240429S0123.fits
         ../mef_20240429/N20240429S0124.fits
         ../mef_20240429/N20240429S0125.fits
         ../mef_20240429/N20240429S0126.fits
         ../mef_20240429/N20240429S0127.fits
         ../mef_20240429/N20240429S0128.fits
         ../mef_20240429/N20240429S0188.fits
         ../mef_20240429/N20240429S0189.fits
         ../mef_20240429/N20240429S0190.fits
         ../mef_20240429/N20240429S0191.fits
         ../mef_20240429/N20240429S0192.fits
         ../mef_20240429/N20240429S0193.fits
         ../mef_20240429/N20240429S0194.fits
         ../mef_20240429/N20240429S0195.fits
         ../mef_20240429/N20240429S0196.fits
         ../mef_20240429/N20240429S0197.fits
         ../mef_20240429/N20240429S0204.fits
         ../mef_20240429/N20240429S0365.fits
         ../mef_20240429/N20240429S0366.fits
         ../mef_20240429/N20240429S0367.fits
         ../mef_20240429/N20240429S0368.fits
         ../mef_20240429/N20240429S0369.fits
         ../mef_20240429/N20240429S0370.fits
         ../mef_20240429/N20240429S0371.fits
         ../mef_20240429/N20240429S0372.fits
         ../mef_20240429/N20240429S0373.fits
         ../mef_20240429/N20240429S0374.fits
         ../mef_20240429/N20240429S0375.fits
         ../mef_20240429/N20240429S0376.fits
         ../mef_20240429/N20240429S0377.fits
         ../mef_20240429/N20240429S0378.fits
         ../mef_20240429/N20240429S0379.fits
         ../mef_20240429/N20240429S0380.fits
         ../mef_20240429/N20240429S0381.fits
         ../mef_20240429/N20240429S0382.fits
         ../mef_20240429/N20240429S0383.fits
         ../mef_20240429/N20240429S0384.fits

.. container:: cell

   .. code:: python

      %reduce_ig @list_of_bundles.txt

   .. container:: cell-output cell-output-stdout

      ::


                     --- reduce v4.1.0 ---

         Running on Python 3.12.12
         All submitted files appear valid:
         ../mef_20240429/N20240429S0120.fits ... ../mef_20240429/N20240429S0384.fits, 40 files submitted.
         ================================================================================
         RECIPE: processBundle
         ================================================================================
            PRIMITIVE: splitBundle
            ----------------------
            Splitting N20240429S0120.fits
            Splitting N20240429S0121.fits
            Splitting N20240429S0122.fits
            Splitting N20240429S0123.fits
            Splitting N20240429S0124.fits
            Splitting N20240429S0125.fits
            Splitting N20240429S0126.fits
            Splitting N20240429S0127.fits
            Splitting N20240429S0128.fits
            Splitting N20240429S0188.fits
            Splitting N20240429S0189.fits
            Splitting N20240429S0190.fits
            Splitting N20240429S0191.fits
            Splitting N20240429S0192.fits
            Splitting N20240429S0193.fits
            Splitting N20240429S0194.fits
            Splitting N20240429S0195.fits
            Splitting N20240429S0196.fits
            Splitting N20240429S0197.fits
            Splitting N20240429S0204.fits
            Splitting N20240429S0365.fits
            Splitting N20240429S0366.fits
            Splitting N20240429S0367.fits
            Splitting N20240429S0368.fits
            Splitting N20240429S0369.fits
            Splitting N20240429S0370.fits
            Splitting N20240429S0371.fits
            Splitting N20240429S0372.fits
            Splitting N20240429S0373.fits
            Splitting N20240429S0374.fits
            Splitting N20240429S0375.fits
            Splitting N20240429S0376.fits
            Splitting N20240429S0377.fits
            Splitting N20240429S0378.fits
            Splitting N20240429S0379.fits
            Splitting N20240429S0380.fits
            Splitting N20240429S0381.fits
            Splitting N20240429S0382.fits
            Splitting N20240429S0383.fits
            Splitting N20240429S0384.fits
            .
             Wrote N20240429S0122_H.fits in output directory
             Wrote N20240429S0123_H.fits in output directory
             Wrote N20240429S0124_H.fits in output directory
             Wrote N20240429S0125_H.fits in output directory
             Wrote N20240429S0126_H.fits in output directory
             Wrote N20240429S0127_H.fits in output directory
             Wrote N20240429S0128_H.fits in output directory
             Wrote N20240429S0190_H.fits in output directory
             Wrote N20240429S0191_H.fits in output directory
             Wrote N20240429S0192_H.fits in output directory
             Wrote N20240429S0193_H.fits in output directory
             Wrote N20240429S0194_H.fits in output directory
             Wrote N20240429S0195_H.fits in output directory
             Wrote N20240429S0196_H.fits in output directory
             Wrote N20240429S0197_H.fits in output directory
             Wrote N20240429S0204_H.fits in output directory
             Wrote N20240429S0365_H.fits in output directory
             Wrote N20240429S0366_H.fits in output directory
             Wrote N20240429S0367_H.fits in output directory
             Wrote N20240429S0368_H.fits in output directory
             Wrote N20240429S0369_H.fits in output directory
             Wrote N20240429S0370_H.fits in output directory
             Wrote N20240429S0371_H.fits in output directory
             Wrote N20240429S0372_H.fits in output directory
             Wrote N20240429S0373_H.fits in output directory
             Wrote N20240429S0374_H.fits in output directory
             Wrote N20240429S0375_H.fits in output directory
             Wrote N20240429S0376_H.fits in output directory
             Wrote N20240429S0377_H.fits in output directory
             Wrote N20240429S0378_H.fits in output directory
             Wrote N20240429S0379_H.fits in output directory
             Wrote N20240429S0380_H.fits in output directory
             Wrote N20240429S0381_H.fits in output directory
             Wrote N20240429S0382_H.fits in output directory
             Wrote N20240429S0383_H.fits in output directory
             Wrote N20240429S0384_H.fits in output directory
             Wrote N20240429S0122_K.fits in output directory
             Wrote N20240429S0123_K.fits in output directory
             Wrote N20240429S0124_K.fits in output directory
             Wrote N20240429S0125_K.fits in output directory
             Wrote N20240429S0126_K.fits in output directory
             Wrote N20240429S0127_K.fits in output directory
             Wrote N20240429S0128_K.fits in output directory
             Wrote N20240429S0190_K.fits in output directory
             Wrote N20240429S0191_K.fits in output directory
             Wrote N20240429S0192_K.fits in output directory
             Wrote N20240429S0193_K.fits in output directory
             Wrote N20240429S0194_K.fits in output directory
             Wrote N20240429S0195_K.fits in output directory
             Wrote N20240429S0196_K.fits in output directory
             Wrote N20240429S0197_K.fits in output directory
             Wrote N20240429S0204_K.fits in output directory
             Wrote N20240429S0365_K.fits in output directory
             Wrote N20240429S0366_K.fits in output directory
             Wrote N20240429S0367_K.fits in output directory
             Wrote N20240429S0368_K.fits in output directory
             Wrote N20240429S0369_K.fits in output directory
             Wrote N20240429S0370_K.fits in output directory
             Wrote N20240429S0371_K.fits in output directory
             Wrote N20240429S0372_K.fits in output directory
             Wrote N20240429S0373_K.fits in output directory
             Wrote N20240429S0374_K.fits in output directory
             Wrote N20240429S0375_K.fits in output directory
             Wrote N20240429S0376_K.fits in output directory
             Wrote N20240429S0377_K.fits in output directory
             Wrote N20240429S0378_K.fits in output directory
             Wrote N20240429S0379_K.fits in output directory
             Wrote N20240429S0380_K.fits in output directory
             Wrote N20240429S0381_K.fits in output directory
             Wrote N20240429S0382_K.fits in output directory
             Wrote N20240429S0383_K.fits in output directory
             Wrote N20240429S0384_K.fits in output directory
         reduce completed successfully.

.. container:: cell

   .. code:: python

      %cd -

   .. container:: cell-output cell-output-stdout

      ::

         /home/jjlee/git_personal/IGRINSDR/test_i2

.. container:: cell

   .. code:: python

      # This require igrinsdr_helper to be installed (pip install igrinsdr-helper)
      from igrinsdr_helper.igrinsdr_tree import get_ad_tree
      from pathlib import Path

      get_ad_tree(Path("./unbundled_20240429").glob("N*_H.fits"))

   .. container:: cell-output cell-output-display

      ::

         Tree(nodes=(Node(icon_style='success', name="'SPECT UNPREPARED NORTH IGRINS GEMINI H IGRINS-2'", nodes=(Node(i…

Running REDUCE
==============

1. FLAT
2. SKY
3. STANDARD
4. TARGET

FLAT
====

Then create a file lising the FALT image from the fixed fits files.

   dataselect_ig –tags FLAT,H unbundled_20240429/N*_H.fits -o
   list_of_flat_h.txt

For now, you need to make a badpixel file using the flat images. For
this, you will the reduce with a specific recipe name.

   reduce_ig @list_of_flat_h.txt -r makeProcessedBPM

Run reduce_ig with the name of the created badpixel file as a parameter.

   reduce_ig @list_of_flat_h.txt -p
   user_bpm=SDCH_20240429_0365_hotpixel.fits

.. container:: cell

   .. code:: python

      %dataselect_ig --tags FLAT,H unbundled_20240429/N*_H.fits -o list_of_flat_h.txt

.. container:: cell

   .. code:: python

      !cat list_of_flat_h.txt

   .. container:: cell-output cell-output-stdout

      ::

         # Includes tags: ['FLAT', 'H']
         # Excludes tags: []
         # Descriptor expression: None
         unbundled_20240429/N20240429S0365_H.fits
         unbundled_20240429/N20240429S0366_H.fits
         unbundled_20240429/N20240429S0367_H.fits
         unbundled_20240429/N20240429S0368_H.fits
         unbundled_20240429/N20240429S0369_H.fits
         unbundled_20240429/N20240429S0370_H.fits
         unbundled_20240429/N20240429S0371_H.fits
         unbundled_20240429/N20240429S0372_H.fits
         unbundled_20240429/N20240429S0373_H.fits
         unbundled_20240429/N20240429S0374_H.fits
         unbundled_20240429/N20240429S0375_H.fits
         unbundled_20240429/N20240429S0376_H.fits
         unbundled_20240429/N20240429S0377_H.fits
         unbundled_20240429/N20240429S0378_H.fits
         unbundled_20240429/N20240429S0379_H.fits
         unbundled_20240429/N20240429S0380_H.fits
         unbundled_20240429/N20240429S0381_H.fits
         unbundled_20240429/N20240429S0382_H.fits
         unbundled_20240429/N20240429S0383_H.fits
         unbundled_20240429/N20240429S0384_H.fits

.. container:: cell

   .. code:: python

      %reduce_ig @list_of_flat_h.txt -r makeProcessedBPM

   .. container:: cell-output cell-output-stdout

      ::


                     --- reduce v4.1.0 ---

         Running on Python 3.12.12
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
               .
            .
             Wrote N20240429S0365_H_badpixel.fits in output directory
         reduce completed successfully.

.. container:: cell

   .. code:: python

      %reduce_ig @list_of_flat_h.txt -p user_bpm=N20240429S0365_H_badpixel.fits

   .. container:: cell-output cell-output-stdout

      ::


                     --- reduce v4.1.0 ---

         Running on Python 3.12.12
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
         /home/jjlee/miniforge3/envs/dragons41/lib/python3.12/site-packages/igrinsdr/igrins/procedures/normalize_flat.py:121: RuntimeWarning: All-NaN slice encountered
           s = np.nanmedian(dn,
            .
            PRIMITIVE: thresholdFlatfield
            -----------------------------
            .
            PRIMITIVE: storeProcessedFlat
            -----------------------------
               PRIMITIVE: storeCalibration
               ---------------------------
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
         Tags: CAL FLAT GCALFLAT GEMINI H IGRINS IGRINS-2 NORTH PREPARED PROCESSED SPECT

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

      %matplotlib inline

.. container:: cell

   .. code:: python

      plt.figure()
      plt.imshow(np.ma.array(ad_flat[0].data, mask=ad_flat[0].mask).filled(np.nan), vmin=0.8, vmax=1.2, origin="lower")

   .. container:: cell-output cell-output-display

      |image1|

SKY
===

(this recipe is a combined version of register-sky and wvlsol-sky from
the PLP)

Create a file containg sky frames.

   dataselect_ig –tags SKY,H unbundled_20240429/N*_H.fits -o
   list_of_sky_h.txt

.. container:: alert alert-info

   CALDB support is not properly integrated yet. You need to explicitly
   speicy calibrations file with the “–user_cal” options.

We will run reduce, but we need to explicitly set the calibration file.

   reduce_ig @list_of_sky_h.txt –user_cal
   processed_flat:SDCH_20240429_0375_flat.fits

.. container:: cell

   .. code:: python

      %dataselect_ig --tags SKY,H unbundled_20240429/N*_H.fits -o list_of_sky_h.txt

.. container:: cell

   .. code:: python

      !cat list_of_sky_h.txt

   .. container:: cell-output cell-output-stdout

      ::

         # Includes tags: ['SKY', 'H']
         # Excludes tags: []
         # Descriptor expression: None
         unbundled_20240429/N20240429S0204_H.fits

.. container:: cell

   .. code:: python

      %reduce_ig @list_of_sky_h.txt --user_cal processed_flat:N20240429S0375_H_flat.fits

   .. container:: cell-output cell-output-stdout

      ::


                     --- reduce v4.1.0 ---

         Running on Python 3.12.12
         All submitted files appear valid:
         unbundled_20240429/N20240429S0204_H.fits
         Manually assigned N20240429S0375_H_flat.fits as processed_flat
         ================================================================================
         RECIPE: makeProcessedArc
         ================================================================================
            PRIMITIVE: fixIgrinsHeader
            --------------------------
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
                  .
                  PRIMITIVE: standardizeInstrumentHeaders
                  ---------------------------------------
                  .
               .
               PRIMITIVE: standardizeWCS
               -------------------------
               WARNING - N20240429S0204_H_observatoryHeadersStandardized.fits (and maybe other files) do not have detector offsets. Cannot check/fix WCS.
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
            PRIMITIVE: readoutPatternCorrectSky
            -----------------------------------
            .
            PRIMITIVE: ADUToElectrons
            -------------------------
            Converting N20240429S0204_H_rpc.fits from ADU to electrons by multiplying by the gain
            .
            PRIMITIVE: streamFirstFrame
            ---------------------------
            .
            PRIMITIVE: stackFrames
            ----------------------
            No stacking will be performed, since at least two input AstroData objects are required for stackFrames
            .
            PRIMITIVE: setReferenceFrame
            ----------------------------
         Sky frames exp time > 30 s.  Using the first frame.
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
         /home/jjlee/miniforge3/envs/dragons41/lib/python3.12/site-packages/igrinsdr/igrins/primitives_igrins.py:1582: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
           df_ref_data = df_ref_data0.set_index("gid")[msk].reset_index()
            .
            PRIMITIVE: volumeFit
            --------------------
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
         Tags: ARC CAL GEMINI H IGRINS IGRINS-2 NORTH PREPARED PROCESSED SIDEREAL SPECT

         Pixels Extensions
         Index  Content                  Type              Dimensions     Format
         [ 0]   science                  NDAstroData       (2048, 2048)   float32
                   .variance             ADVarianceUncerta (2048, 2048)   float32
                   .mask                 ndarray           (2048, 2048)   uint16
                   .FLEXCORR             ndarray           (2048, 2048)   float32
                   .LINEFIT              Table             (755, 10)      n/a
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

      |image2|

.. container:: cell

   .. code:: python

      plt.figure()
      plt.imshow(ad_sky[0].SLITPOSMAP, origin="lower")

   .. container:: cell-output cell-output-display

      |image3|

.. container:: cell

   .. code:: python

      ad_sky[0].WVLSOL

   .. container:: cell-output cell-output-display

      .. raw:: html

         <div><i>Table length=27</i>
         <table id="table130246895758048" class="table-striped table-bordered table-condensed">
         <thead><tr><th>orders</th><th>wavelengths</th></tr></thead>
         <thead><tr><th>int64</th><th>float64[2048]</th></tr></thead>
         <tr><td>98</td><td>1.8111782542108175 .. 1.8360355144088847</td></tr>
         <tr><td>99</td><td>1.7933426895060407 .. 1.8179597713646614</td></tr>
         <tr><td>100</td><td>1.7758681385798152 .. 1.8002494404830163</td></tr>
         <tr><td>101</td><td>1.7587439812773917 .. 1.782893672643976</td></tr>
         <tr><td>102</td><td>1.7419600139206783 .. 1.7658820441832508</td></tr>
         <tr><td>103</td><td>1.7255064290909259 .. 1.7492045362390494</td></tr>
         <tr><td>104</td><td>1.7093737965777989 .. 1.7328515152904171</td></tr>
         <tr><td>105</td><td>1.6935530454170697 .. 1.7168137148076754</td></tr>
         <tr><td>106</td><td>1.6780354469450487 .. 1.7010822179415082</td></tr>
         <tr><td>...</td><td>...</td></tr>
         <tr><td>115</td><td>1.5507289869980991 .. 1.5719646536388854</td></tr>
         <tr><td>116</td><td>1.5378277608000863 .. 1.5588720877995714</td></tr>
         <tr><td>117</td><td>1.5251522576605656 .. 1.5460067266025135</td></tr>
         <tr><td>118</td><td>1.5126968270416608 .. 1.5333627976870645</td></tr>
         <tr><td>119</td><td>1.5004560083395406 .. 1.5209347227215062</td></tr>
         <tr><td>120</td><td>1.4884245229705029 .. 1.5087171093185106</td></tr>
         <tr><td>121</td><td>1.4765972668494782 .. 1.4967047433514884</td></tr>
         <tr><td>122</td><td>1.4649693032384512 .. 1.4848925816488199</td></tr>
         <tr><td>123</td><td>1.4535358559437301 .. 1.4732757450444671</td></tr>
         <tr><td>124</td><td>1.4422923028423884 .. 1.4618495117648447</td></tr>
         </table></div>

A0V
===

Let’s do telluric standar star. We will select images from
observation_id of ‘GN-2024A-ENG-142-261’.

   dataselect_ig –tags STANDARD,H unbundled_20240429/N*_H.fits -o
   list_of_std_h.txt –expr “observation_id==‘GN-2024A-ENG-142-120’”

Again, we need to explicitly specify calibration files.

   reduce_ig @list_of_std_h.txt –user_cal
   processed_flat:N20240429S0375_H_flat
   processed_arc:N20240429S0204_H_arc.fits

.. container:: cell

   .. code:: python

      %dataselect_ig --tags STANDARD,H unbundled_20240429/N*_H.fits -o list_of_std_h.txt --expr "observation_id=='GN-2024A-ENG-142-120'"

.. container:: cell

   .. code:: python

      !cat list_of_std_h.txt

   .. container:: cell-output cell-output-stdout

      ::

         # Includes tags: ['STANDARD', 'H']
         # Excludes tags: []
         # Descriptor expression: observation_id=='GN-2024A-ENG-142-120'
         unbundled_20240429/N20240429S0194_H.fits
         unbundled_20240429/N20240429S0195_H.fits
         unbundled_20240429/N20240429S0196_H.fits
         unbundled_20240429/N20240429S0197_H.fits

.. container:: cell

   .. code:: python

      %reduce_ig @list_of_std_h.txt --user_cal processed_flat:N20240429S0375_H_flat.fits processed_arc:N20240429S0204_H_arc.fits

   .. container:: cell-output cell-output-stdout

      ::


                     --- reduce v4.1.0 ---

         Running on Python 3.12.12
         All submitted files appear valid:
         unbundled_20240429/N20240429S0194_H.fits ... unbundled_20240429/N20240429S0197_H.fits, 4 files submitted.
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
                  .
                  PRIMITIVE: standardizeInstrumentHeaders
                  ---------------------------------------
                  .
               .
               PRIMITIVE: standardizeWCS
               -------------------------
               WARNING - N20240429S0194_H_observatoryHeadersStandardized.fits (and maybe other files) do not have detector offsets. Cannot check/fix WCS.
               .
            .
            PRIMITIVE: addDQ
            ----------------
            No BPMs found for N20240429S0194_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0195_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0196_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0197_H_prepared.fits and none supplied by the user.
            
            .
            PRIMITIVE: addVAR
            -----------------
            Adding the read noise component and the Poisson noise component of the variance
            .
            PRIMITIVE: ADUToElectrons
            -------------------------
            Converting N20240429S0194_H_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0195_H_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0196_H_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0197_H_varAdded.fits from ADU to electrons by multiplying by the gain
            .
            PRIMITIVE: makeAB
            -----------------
         #### stackFrames!!
               PRIMITIVE: stackFrames
               ----------------------
               Combining 2 inputs with mean and sigclip rejection
               Combining images.
               
               .
         #### stackFrames!!
               PRIMITIVE: stackFrames
               ----------------------
               Combining 2 inputs with mean and sigclip rejection
               Combining images.
               
               .
         /home/jjlee/miniforge3/envs/dragons41/lib/python3.12/site-packages/astrodata/nddata.py:32: RuntimeWarning: Negative variance values found. Setting to zero.
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
             Wrote N20240429S0194_H_spec1d.fits in output directory
         reduce completed successfully.

.. container:: cell

   .. code:: python

      ad_std = astrodata.open("N20240429S0194_H_spec1d.fits")
      ad_std.info()

   .. container:: cell-output cell-output-stdout

      ::

         Filename: N20240429S0194_H_spec1d.fits
         Tags: CAL GEMINI H IGRINS IGRINS-2 NORTH PREPARED SIDEREAL SPECT STANDARD

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

      |image4|

.. container:: cell

   .. code:: python

      fig, ax = plt.subplots(figsize=(12, 4))
      for w, s in zip(ad_std[0].WAVELENGTHS, ad_std[0].data):
          ax.plot(w[6:-6], s[6:-6])

   .. container:: cell-output cell-output-display

      |image5|

.. container:: cell

   .. code:: python

      ax.set_xlim(1.7, 1.75)
      ax.set_ylim(-4000, 28000)
      fig

   .. container:: cell-output cell-output-display

      |image6|

.. container:: cell

   .. code:: python

      ad_std_2dspec = astrodata.open("N20240429S0194_H_spec2d.fits")
      ad_std_2dspec.info()

   .. container:: cell-output cell-output-stdout

      ::

         Filename: N20240429S0194_H_spec2d.fits
         Tags: CAL GEMINI H IGRINS IGRINS-2 NORTH PREPARED SIDEREAL SPECT STANDARD

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

      |image7|

.. container:: cell

   .. code:: python

      ad_std_debug = astrodata.open("N20240429S0194_H_spec_debug.fits")
      ad_std_debug.info()

   .. container:: cell-output cell-output-stdout

      ::

         Filename: N20240429S0194_H_spec_debug.fits
         Tags: CAL GEMINI H IGRINS IGRINS-2 NORTH PREPARED SIDEREAL SPECT STANDARD

         Pixels Extensions
         Index  Content                  Type              Dimensions     Format
         [ 0]   science                  NDAstroData       (2048, 2048)   float64
                   .variance             ADVarianceUncerta (2048, 2048)   float64
                   .mask                 ndarray           (2048, 2048)   uint16
                   .SLITPROFILE_MAP      ndarray           (2048, 2048)   float64
                   .SPEC1D               Table             (27, 5)        n/a
                   .WVLCOR               Table             (4, 2)         n/a

SCIENCE
=======

For science target, select 900078.

   dataselect_ig –tags SCIENCE,H unbundled_20240429/N*_H.fits -o
   list_of_900078_h.txt –expr “object==‘900078’”

Run reduce.

   reduce_ig @list_of_900078_h.txt –user_cal
   processed_flat:N20240429S0375_H_flat.fits
   processed_arc:N20240429S0204_H_arc.fits

.. container:: cell

   .. code:: python

      %dataselect_ig --tags SIDEREAL,H unbundled_20240429/N*_H.fits -o list_of_900078_h.txt --expr "object=='900078'"

.. container:: cell

   .. code:: python

      !cat list_of_900078_h.txt

   .. container:: cell-output cell-output-stdout

      ::

         # Includes tags: ['SIDEREAL', 'H']
         # Excludes tags: []
         # Descriptor expression: object=='900078'
         unbundled_20240429/N20240429S0190_H.fits
         unbundled_20240429/N20240429S0191_H.fits
         unbundled_20240429/N20240429S0192_H.fits
         unbundled_20240429/N20240429S0193_H.fits

.. container:: cell

   .. code:: python

      %reduce_ig @list_of_900078_h.txt --user_cal processed_flat:N20240429S0375_H_flat.fits processed_arc:N20240429S0204_H_arc.fits

   .. container:: cell-output cell-output-stdout

      ::


                     --- reduce v4.1.0 ---

         Running on Python 3.12.12
         All submitted files appear valid:
         unbundled_20240429/N20240429S0190_H.fits ... unbundled_20240429/N20240429S0193_H.fits, 4 files submitted.
         Manually assigned N20240429S0375_H_flat.fits as processed_flat
         Manually assigned N20240429S0204_H_arc.fits as processed_arc
         ================================================================================
         RECIPE: makeTgt
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
                  .
                  PRIMITIVE: standardizeInstrumentHeaders
                  ---------------------------------------
                  .
               .
               PRIMITIVE: standardizeWCS
               -------------------------
               WARNING - N20240429S0190_H_observatoryHeadersStandardized.fits (and maybe other files) do not have detector offsets. Cannot check/fix WCS.
               .
            .
            PRIMITIVE: addDQ
            ----------------
            No BPMs found for N20240429S0190_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0191_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0192_H_prepared.fits and none supplied by the user.
            
            No BPMs found for N20240429S0193_H_prepared.fits and none supplied by the user.
            
            .
            PRIMITIVE: addVAR
            -----------------
            Adding the read noise component and the Poisson noise component of the variance
            .
            PRIMITIVE: ADUToElectrons
            -------------------------
            Converting N20240429S0190_H_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0191_H_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0192_H_varAdded.fits from ADU to electrons by multiplying by the gain
            Converting N20240429S0193_H_varAdded.fits from ADU to electrons by multiplying by the gain
            .
            PRIMITIVE: makeAB
            -----------------
         #### stackFrames!!
               PRIMITIVE: stackFrames
               ----------------------
               Combining 2 inputs with mean and sigclip rejection
               Combining images.
               
               .
         #### stackFrames!!
               PRIMITIVE: stackFrames
               ----------------------
               Combining 2 inputs with mean and sigclip rejection
               Combining images.
               
               .
         /home/jjlee/miniforge3/envs/dragons41/lib/python3.12/site-packages/astrodata/nddata.py:32: RuntimeWarning: Negative variance values found. Setting to zero.
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
             Wrote N20240429S0190_H_spec1d.fits in output directory
         reduce completed successfully.

.. container:: cell

   .. code:: python

      ad_tgt = astrodata.open("N20240429S0190_H_spec1d.fits")
      ad_tgt.info()

   .. container:: cell-output cell-output-stdout

      ::

         Filename: N20240429S0190_H_spec1d.fits
         Tags: GEMINI H IGRINS IGRINS-2 NORTH PREPARED SIDEREAL SPECT

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

      |image8|

.. container:: cell

   .. code:: python

      fig, ax = plt.subplots(figsize=(12, 4))
      for w, s, t in zip(ad_std[0].WAVELENGTHS, ad_std[0].data, ad_tgt[0].data):
          ax.plot(w, t/s)
      ax.set_ylim(0., 0.7)

   .. container:: cell-output cell-output-display

      |image9|

.. container:: cell

   .. code:: python

      ax.set_xlim(1.72, 1.75)
      ax.set_ylim(0., 0.2)
      fig

   .. container:: cell-output cell-output-display

      |image10|

.. |image1| image:: IGRINSDR_users_manual_files/figure-rst/cell-17-output-1.png
.. |image2| image:: IGRINSDR_users_manual_files/figure-rst/cell-22-output-1.png
.. |image3| image:: IGRINSDR_users_manual_files/figure-rst/cell-23-output-1.png
.. |image4| image:: IGRINSDR_users_manual_files/figure-rst/cell-29-output-1.png
.. |image5| image:: IGRINSDR_users_manual_files/figure-rst/cell-30-output-1.png
.. |image6| image:: IGRINSDR_users_manual_files/figure-rst/cell-31-output-1.png
.. |image7| image:: IGRINSDR_users_manual_files/figure-rst/cell-33-output-1.png
.. |image8| image:: IGRINSDR_users_manual_files/figure-rst/cell-39-output-1.png
.. |image9| image:: IGRINSDR_users_manual_files/figure-rst/cell-40-output-1.png
.. |image10| image:: IGRINSDR_users_manual_files/figure-rst/cell-41-output-1.png
