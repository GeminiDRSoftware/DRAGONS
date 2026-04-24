SKY Recipe
==========

This document describes the SKY recipe available in the IGRINS data reduction pipeline. The SKY recipe is used for processing sky/arc lamp observations, which are essential for wavelength calibration.

Available Recipe
----------------
- `[makeProcessedArc](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/recipes/sq/recipe_SKY.py:8:0-61:10)` (default)

makeProcessedArc
----------------
The default recipe for processing sky/arc lamp observations. This recipe performs the standardization and corrections needed to convert raw input sky/arc lamp images into wavelength-calibrated spectra. The output processed arc is stored on disk and its information is added to the calibration database.

**Processing Steps:**

1. **Header Fixing and Preparation**
   - Fixes IGRINS-specific FITS headers
   - Prepares the data for processing

2. **Basic Corrections**
   - Adds data quality (DQ) information
   - Adds variance planes including read noise
   - Applies readout pattern correction specific to sky observations
   - Converts ADU to electron counts

3. **Frame Processing**
   - Streams and stacks the first frame
   - Sets the reference frame for further processing

4. **Spectral Extraction**
   - Performs simple spectral extraction
   - Identifies spectral orders
   - Identifies spectral lines for wavelength calibration

5. **Wavelength Calibration**
   - Gets initial wavelength solution
   - Performs multi-line identification
   - Fits volume data for accurate wavelength calibration

6. **Final Processing**
   - Creates spectral maps
   - Attaches WCS (World Coordinate System) table
   - Stores the processed arc

**Primitives Used:**

- `[fixIgrinsHeader()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:1017:4-1037:23)`: Fixes IGRINS-specific FITS headers
- ``prepare()``: Prepares the data for processing
- ``addDQ()``: Adds data quality information
- ``addVAR()``: Adds variance planes including read noise
- `[readoutPatternCorrectSky()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:1039:4-1053:23)`: Corrects readout pattern specific to sky observations
- ``ADUToElectrons()``: Converts ADU to electron counts
- `[streamFirstFrame()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:730:4-733:23)`: Streams the first frame
- ``stackFrames()``: Stacks multiple frames
- `[setReferenceFrame()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:735:4-751:23)`: Sets the reference frame
- `[extractSimpleSpec()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:1118:4-1178:23)`: Performs simple spectral extraction
- `[identifyOrders()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:1181:4-1255:23)`: Identifies spectral orders
- `[identifyLines()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:1258:4-1322:23)`: Identifies spectral lines
- `[getInitialWvlsol()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:1324:4-1403:23)`: Gets initial wavelength solution
- `[extractSpectraMulti()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:1405:4-1470:23)`: Extracts spectra from multiple orders
- `[identifyMultiline()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:1472:4-1623:23)`: Identifies multiple spectral lines
- `[volumeFit()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:1703:4-1757:23)`: Fits volume data for wavelength calibration
- `[makeSpectralMaps()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:1807:4-1908:23)`: Creates spectral maps
- `[attachWatTable()](cci:1://file:///home/jjlee/git_personal/IGRINSDR/src/igrinsdr/igrins/primitives_igrins.py:1721:4-1731:23)`: Attaches WCS (World Coordinate System) table
- ``storeProcessedArc()``: Stores the processed arc

Usage Example
-------------

To use the default recipe:

.. code-block:: python

    from igrins import IGRINS
    p = IGRINS(files)
    p.reduce('SKY')

Output Files
------------
- Processed arc files (``*_arc.fits``)
- Wavelength solution files
- Spectral maps
- Calibration database entries

Notes
-----
- The recipe requires sky/arc lamp observations
- The output is used for wavelength calibration of science data
- The recipe is typically run once per observing run or when the instrument configuration changes
- The output is stored in the calibration database for use in reducing science data