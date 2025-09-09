FLAT Recipe
===========

This document describes the FLAT recipes available in the IGRINS data reduction pipeline. The FLAT recipes are used for processing flat field calibration frames, which are essential for correcting pixel-to-pixel sensitivity variations in the detector.

Available Recipes
-----------------

1. `[makeProcessedFlat](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/recipes/sq/recipes_FLAT.py:41:0-92:10)` (default)
2. `[estimateNoise](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:754:4-780:23)`
3. `[makeProcessedBPM](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/recipes/sq/recipes_FLAT.py:101:0-124:10)`
4. `[makeTestBadpix](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/recipes/sq/recipes_FLAT.py:127:0-151:10)`

makeProcessedFlat
-----------------
The default recipe for processing flat field images. It creates a processed flat field by:

1. Separating lamp-on and lamp-off images
2. Stacking each set
3. Subtracting lamp-off from lamp-on
4. Normalizing the flat field by order
5. Determining slit edges and masking unilluminated pixels

**Primitives Used:**

- ``prepare()``: Initial preparation of the input frames
- `[readoutPatternCorrectFlatOff()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:1056:4-1080:23)`: Corrects readout pattern in flat-off images
- ``addDQ()``: Adds data quality information
- ``addVAR()``: Adds variance planes including read noise and Poisson noise
- ``ADUToElectrons()``: Converts ADU to electron counts
- ``makeLampFlat()``: Processes lamp-on and lamp-off frames
- `[determineSlitEdges()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:845:4-885:23)`: Identifies slit edges
- `[maskBeyondSlit()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:887:4-932:23)`: Masks unilluminated pixels
- `[normalizeFlat()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:934:4-1015:23)`: Normalizes the flat field
- ``thresholdFlatfield()``: Applies thresholding to the flat field
- ``storeProcessedFlat()``: Saves the processed flat

estimateNoise
-------------
Analyzes the readout pattern noise in flat-off images.

**Primitives Used:**

- `[selectFrame()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:681:4-687:24)`: Selects specific frame types (OFF)
- ``prepare()``: Initial preparation
- `[streamPatternCorrected()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:689:4-724:23)`: Creates pattern-corrected images
- `[estimateNoise()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:754:4-780:23)`: Estimates noise characteristics
- `[selectStream()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:782:4-784:40)`: Selects specific data streams
- ``stackFlats()``: Stacks flat field images
- `[addNoiseTable()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:786:4-801:23)`: Adds noise characteristics to the output
- `[setSuffix()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:803:4-811:23)`: Sets the output file suffix

makeProcessedBPM
----------------
Creates a Bad Pixel Mask (BPM) using flat field data.

**Primitives Used:**

- ``prepare()``: Initial preparation
- ``addDQ()``: Adds data quality information
- `[readoutPatternCorrectFlatOff()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:1056:4-1080:23)`: Corrects flat-off images
- `[readoutPatternCorrectFlatOn()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:1082:4-1094:23)`: Corrects flat-on images
- ``selectFromInputs()``: Selects specific input types
- ``stackFrames()``: Stacks selected frames
- `[makeIgrinsBPM()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:2247:4-2278:27)`: Creates the bad pixel mask
- ``storeBPM()``: Saves the BPM

makeTestBadpix
--------------
Test recipe for generating and examining bad pixel masks.

**Primitives Used:**

- ``prepare()``: Initial preparation
- `[readoutPatternCorrectFlatOff()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:1056:4-1080:23)`: Corrects flat-off images
- ``addDQ()``: Adds data quality information
- ``addVAR()``: Adds variance planes
- `[fixIgrinsHeader()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:1017:4-1037:23)`: Fixes FITS headers
- ``ADUToElectrons()``: Converts ADU to electron counts
- ``selectFromInputs()``: Selects specific input types
- ``stackFrames()``: Stacks selected frames

Usage Examples
-------------

To use the default recipe:

.. code-block:: python

    from igrins import IGRINS
    p = IGRINS(files)
    p.reduce('FLAT')

To use a specific recipe:

.. code-block:: python

    from igrins import IGRINS
    p = IGRINS(files)
    p.reduce('FLAT', recipe='estimateNoise')

Output Files
------------
- Processed flat fields (``*_flat.fits``)
- Bad pixel masks (``*_bpm.fits`` when using makeProcessedBPM)
- Noise analysis files (``*_pattern_noise.fits`` when using estimateNoise)

Notes
-----
- The recipes require both lamp-on and lamp-off flat field images
- The default recipe (makeProcessedFlat) is the most commonly used for standard data reduction
- The BPM recipe (makeProcessedBPM) is typically run once per observing run