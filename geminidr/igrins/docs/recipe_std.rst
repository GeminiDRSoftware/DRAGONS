STD Recipe
==========

The STD (Standard Star) recipe is designed for processing standard star observations with IGRINS. This recipe performs the reduction of standard star data to extract calibrated spectra, which can then be used for flux calibration of science targets.

Overview
--------

The STD recipe processes standard star observations through the following main steps:

1. **Data Preparation and Quality Control**:
   - Verifies the presence of required calibration files (processed flat and arc)
   - Adds data quality (DQ) information
   - Adds variance planes including read noise and Poisson noise

2. **Detector Calibration**:
   - Converts ADU to electron counts using the detector gain
   - Handles detector-specific corrections

3. **Spectral Extraction**:
   - Creates A-B pairs (if nod-to-slit mode is used)
   - Estimates the slit profile
   - Extracts 1D spectra from the 2D frames

4. **Output**:
   - Saves the 2D extracted spectra
   - Generates debug images for quality assessment

Primitives Used
---------------

The STD recipe utilizes the following primitives from the IGRINS pipeline:

- ``checkCALDB(caltypes=["processed_flat", "processed_arc"])``
  - Verifies that required calibration files (flat and arc) are available in the calibration database

- ``prepare(require_wcs=False)``
  - Performs initial frame preparation
  - The `require_wcs=False` parameter indicates that WCS information is not strictly required at this stage

- ``addDQ()``
  - Adds Data Quality (DQ) extension to the data
  - Flags bad pixels and other detector artifacts

- ``addVAR(read_noise=True, poisson_noise=True)``
  - Adds variance planes to the data
  - Includes both read noise and Poisson noise components

- ``ADUToElectrons()``
  - Converts ADU (Analog-to-Digital Units) to electron counts
  - Uses the detector gain value from the header

- ``makeAB()``
  - Creates A-B (or B-A) pairs for nod-to-slit observations
  - Performs reference pixel correction

- ``estimateSlitProfile()``
  - Estimates the spatial profile of the slit
  - Used for optimal extraction of the spectrum

- ``extractStellarSpec()``
  - Extracts 1D spectra from 2D frames
  - Can perform both optimal and box extraction

- ``saveTwodspec()``
  - Saves the 2D extracted spectra to disk

- ``saveDebugImage()``
  - Saves additional diagnostic images for quality control

Usage Notes
-----------

1. **Input Requirements**:
   - Raw standard star observations
   - Processed flat field (for flat fielding)
   - Processed arc frames (for wavelength calibration)

2. **Output Products**:
   - Extracted 1D spectra
   - 2D extracted spectra
   - Diagnostic images for quality assessment

3. **Header Keywords**:
   - The recipe expects certain header keywords to be present
   - These include detector gain, read noise, and other instrument parameters

4. **Common Issues**:
   - Missing calibration files will cause the recipe to fail
   - Poor signal-to-noise in the standard star may affect the quality of the extracted spectrum

Example
-------

To process standard star observations using the STD recipe:

.. code-block:: python

    from igrinsdr.igrins import IGRINS
    from igrinsdr.igrins.recipes import recipe_std

    # Initialize the pipeline
    p = IGRINS(files='path/to/standard_obs*.fits')

    # Run the STD recipe
    recipe_std.makeStd(p)

This will process the standard star observations and produce the necessary output files for flux calibration of science targets.