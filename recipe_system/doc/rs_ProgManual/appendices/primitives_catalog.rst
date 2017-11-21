.. primitives_catalog.rst
.. include interfaces

Primitive Class Catalog
=======================
The following catalog lists all currently defined classes under *gemindr*.
In conjunction with the class name, this appendix also lists the public
methods explicitly defined in that class (as opposed to inherited methods).
If "Public methods" indicates "Not Implemented," then *no* methods are 
explicitly defined on the class and only inherited methods are available. 
Classes in parentheses to the right of the class name indicate the superclass 
or classes of the listed class.

Readers can consult the :ref:`diagram on Primitive Class Hierarchy <prmcls>` 
to help visualize how these pieces fit together.

.. _defprims:

Base Class and Primitives
-------------------------
**PrimitivesBASE**

Public methods::

  None

Core Classes and Primitives
---------------------------
The classes and primitives (public methods) listed in this section all
directly subclass PrimitivesBASE.

**Bookkeeping** (PrimitivesBASE)

  **tagset** = None

Public methods::

 addToList(purpose=None)
 clearAllStreams()
 clearStream()
 getList(max_frames=None, purpose=None)
 showInputs()
 showList(purpose='all')
 writeOutputs(clobber=True, prefix='', strip=False, suffix='', outfilename=None)

**CalibDB** (PrimitivesBASE)

  **tagset** = None

Public methods::

  addCalibration()
  getCalibration(caltype=None)
  getProcessedArc()
  getProcessedBias()
  getProcessedDark()
  getProcessedFlat()
  getProcessedFringe()
  getMDF()
  storeCalibration()
  storeProcessedArc(suffix='_arc')
  storeProcessedBias(suffix='_bias')
  storeBPM()
  storeProcessedDark(suffix='_dark')
  storeProcessedFlat(suffix='_flat')
  storeProcessedFringe(suffix='_fringe')

**CCD** (PrimitivesBASE)

  **tagset** = None

Public methods::

 biasCorrect(bias=None, suffix="_biasCorrected")
 overscanCorrect(average="mean", niterate=2, high_reject=3.0, low_reject=3.0, nbiascontam=None,
                 order=None, suffix="_overscanCorrected")
 subtractBias(bias=None, suffix="_biasCorrected")
 subtractOverscan(average="mean", niterate=2, high_reject=3.0, low_reject=3.0, nbiascontam=None,
                  order=None, suffix="_overscanSubtracted")
 trimOverscan(suffix="_overscanTrimmed")


**NearIR** (PrimitivesBASE)

  **tagset** = None

Public methods::

  makeBPM()
  lampOnLampOff()
  separateFlatsDarks()
  separateLampOff()
  stackDarks()
  stackLampOnLampOff()
  subtractLampOnLampOff()


**Photometry** (PrimitivesBASE)

  **tagset** = None

Public methods::

  addReferenceCatalog(radius=0.067, source="gmos", suffix="_refcatAdded")
  detectSources(mask=False, replace_flags=249, set_saturation=False, suffix="_sourcesDetected")

**Preprocess** (PrimitivesBASE)

  **tagset** = None

Public methods::

  addObjectMaskToDQ(suffix="_objectMaskAdded")
  ADUToElectrons(suffix="_ADUToElectrons")
  applyDQPlane(replace_flags=255, replace_value="median", suffix="_dqPlaneApplied")
  associateSky(time=600., distance=3., max_skies=None, use_all=False, suffix="_skyAssociated")
  correctBackgroundToReferenceImage(remove_zero_level=False, suffix="_backgroundCorrected")
  darkCorrect(dark=None, suffix='_darkCorrected')
  divideByFlat(flat=None, suffix="_flatCorrected")
  flatCorrect(flat=None,suffix="_flatCorrected")
  makeSky(max_skies=None)
  nonlinearityCorrect(suffix="_nonlinearityCorrected")
  normalizeFlat(scale="median", suffix="_normalized")
  separateSky(ref_obj="", ref_sky="", frac_FOV=0.9, suffix="_skySeparated")
  skyCorrect()
  subtractDark(dark=None, suffix="_darkCorrected")
  subtractSky(suffix="_skyCorrected")
  subtractSkyBackground(suffix="_skyBackgroundSubtracted")
  thresholdFlatfield(upper=10.0, lower=0.01, suffix="_thresholdFlatfielded")


**Register** (PrimitivesBASE)

  **tagset** = None

Public methods::

  correctWCSToReferenceFrame(method="sources", fallback=None, use_wcs=True, first_pass=10.0,
                             min_sources=3, cull_sources=False, rotate=False, scale=False, 
			     suffix="_wcsCorrected")

  determineAstrometricSolution(full_wcs=None, suffix="_astrometryCorrected")


**Resample** (PrimitivesBASE)

  **tagset** = None

Public methods::

  alignToReferenceFrame(interpolator="nearest", trim_data=False, suffix="_align")

**Spect** (PrimitivesBASE)

  **tagset** = set(["GEMINI", "SPECT"])

Public methods::

  determineWavelengthSolution (Not Yet Implemented)
  extract1DSpectra            (Not Yet Implemented)
  makeFlat                    (Not Yet Implemented)
  rejectCosmicRays            (Not Yet Implemented)
  resampleToLinearCoords      (Not Yet Implemented)
  skyCorrectFromSlit          (Not Yet Implemented)

**Stack** (PrimitivesBASE)

  **tagset** = None

Public methods::

  alignAndStack(check_if_stack=False)
  stackFlats(mask=True,nhigh=1,nlow=1,operation="median",reject_method="minmax",suffix="_stack")
  stackFrames(mask=True,nhigh=1,nlow=1,operation="average",reject_method="avsigclip")
  stackSkyFrames()


**Standardize** (PrimitivesBASE)

  **tagset** = None

Public methods::

  addDQ(bpm=None, illum_mask=False, suffix="_dqAdded")
  addIllumMaskToDQ(mask=None, suffix="_illumMaskAdded")
  addMDF(mdf=None, suffix="_mdfAdded")
  addVAR(read_noise=False, poisson_noise=False, suffix="_varAdded")
  makeIRAFCompatible()
  prepare(suffix="_prepared")
  standardizeHeaders()
  standardizeInstrumentHeaders(suffix="_instrumentHeadersStandardized")
  standardizeObservatoryHeaders(suffix="_observatoryHeadersStandardized")
  standardizeStructure(suffix="_structureStandardized")
  validateData(num_exts=1, repair=False, suffix="_dataValidated")


**Visualize** (PrimitivesBASE)

  **tagset** = None

Public methods::

  display(extname="SCI", frame=1, ignore=False, overlay=None, remove_bias=False, threshold="auto", 
          tile=True, zscale=True)
  mosaicDetectors(tile=False, interpolate_gaps=False, interpolator="linear", suffix="_mosaicked")
  tileArrays(tile_all=False, suffix="_tiled")


Derived Core Classes and Primitives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following *core* classes do not directly subclass PrimitivesBASE, but 
inherit from other core classes that do.

**Image** (Register, Resample)

  **tagset** = set(["IMAGE"])

Public methods::

  fringeCorrect()
  makeFringe(subtract_median_image=None)
  makeFringeFrame(operation="median", reject_method="avsigclip", subtract_median_image=True, 
                  suffix="_fringe")
  scaleByIntensity(suffix="_scaled")
  scaleFringeToScience(science=None, stats_scale=False, suffix="_fringeScaled")
  subtractFringe(fringe=None, suffix="_fringeSubtracted")


Gemini Classes and Primitives
-----------------------------

The Gemini Class does not subclass on PrimitivesBASE directly, but rather, 
inherits *core* classes defined there and others (such as, 'QA'). The Gemini 
class also marks the appearance of the class attribute ``tagset`` that first 
includes tags useful to the PrimitiveMapper search algorithm. I.e., tags 
applicable to real instrument data.

All classes inheriting from Gemini will override the tagset attribute, tuned 
to that class's particular instrument data processing capabilities.

**Gemini** (Standardize, Bookkeeping, Preprocess, Visualize, 
Stack, QA, CalibDB)

  **tagset** = set(["GEMINI"])

Public methods::

  standardizeObservatoryHeaders(suffix="_observatoryHeadersStandardized")

**QA** (PrimitivesBASE)

  **tagset** = set(["GEMINI"])

Public methods::

  measureBG(remove_bias=False, separate_ext=False, suffix='_bgMeasured')
  measureCC(suffix='_ccMeasured')
  measureIQ(remove_bias=False, separate_ext=False, display=False, suffix='_iqMeasured')

F2 Classes and Primitives
^^^^^^^^^^^^^^^^^^^^^^^^^

**F2** (Gemini, NearIR)

  **tagset** = set(["GEMINI", "F2"])

Public methods::

  standardizeInstrumentHeaders(suffix="_instrumentHeadersStandardized")
  standardizeStructure(suffix="_structureStandardized")

**F2Image** (F2, Image, Photometry)

  **tagset** = set(["GEMINI", "F2", "IMAGE"])

Public methods::

  makeLampFlat()

GMOS Classes and Primitives
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**GMOS** (Gemini, CCD)

  **tagset** = set(["GEMINI", "GMOS"])

Public methods::

  mosaicDetectors(tile=False, interpolate_gaps=False, interpolator="linear", suffix="_mosaicked")
  standardizeInstrumentHeaders(suffix="_instrumentHeadersStandardized")
  subtractOverscan(average="mean", niterate=2, high_reject=3.0, low_reject=3.0, nbiascontam=None,
                   order=None, suffix="_overscanSubtracted")
  tileArrays(tile_all=False, suffix="_tiled")

**GMOSIFU** (GMOSSpect, GMOSNodAndShuffle)

  **tagset** = set(["GEMINI", "GMOS", "SPECT", "IFU"])

Public methods::

  Not Implemented

**GMOSImage** (GMOS, Image, Photometry)

  **tagset** = set(["GEMINI", "GMOS", "IMAGE"])

Public methods::

  fringeCorrect()
  makeFringe(subtract_median_image=None)
  makeFringeFrame(operation="median", reject_method="avsigclip", subtract_median_image=True,
                  suffix="_fringe")
  normalizeFlat(scale="median", suffix="_normalized")
  scaleByIntensity(suffix="_scaled")
  scaleFringeToScience(science=None, stats_scale=False, suffix="_fringeScaled")
  stackFlats(mask=True, nhigh=1, nlow=1, operation="median", reject_method="minmax", suffix="_stack")

**GMOSLongslit** (GMOSSpect, GMOSNodAndShuffle)

  **tagset** = set(["GEMINI", "GMOS", "SPECT", "LS"])

Public methods::

  Not Implemented          

**GMOSMOS** (GMOSSpect, GMOSNodAndShuffle)

  **tagset** = set(["GEMINI", "GMOS", "SPECT", "MOS"])

Public methods::

  Not Implemented

**GMOSNodAndShuffle** (Mixin)

  **tagset** = set()

Public methods::

  skyCorrectNodAndShuffle(suffix="_skyCorrected")

**GMOSSpect** (GMOS, Spect)

  **tagset** = set(["GEMINI", "GMOS", "SPECT"])

Public methods::

  findAcquisitionSlits(suffix="_acqSlitsAdded")

GNIRS Classes and Primitives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**GNIRS** (Gemini, NearIR)

  **tagset** = set(["GEMINI", "GNIRS"])

Public methods::

  standardizeInstrumentHeaders(suffix="_instrumentHeadersStandardized")

**GNIRSImage** (GNIRS, Image, Photometry)

  **tagset** = set(["GEMINI", "GNIRS", "IMAGE"])

Public methods::

  addIllumMaskToDQ(mask=None, suffix="_illumMaskAdded")


GSAOI Classes and Primitives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**GSAOI** (Gemini, NearIR)

  **tagset** = set(["GEMINI", "GSAOI"])

Public methods::

  standardizeInstrumentHeaders(suffix="_instrumentHeadersStandardized")
  tileArrays(tile_all=True, suffix="_tiled")

**GSAOIImage** (GSAOI, Image, Photometry)

  **tagset** = set(["GEMINI", "GSAOI", "IMAGE"])

Public methods::

  makeLampFlat()

NIRI Classes and Primitives
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**NIRI** (Gemini, NearIR)

  **tagset** = set(["GEMINI", "NIRI"])

Public methods::

  nonlinearityCorrect(suffix="_nonlinearityCorrected")
  standardizeInstrumentHeaders(suffix="_instrumentHeadersStandardized")

**NIRIImage** (NIRI, Image, Photometry)

  **tagset** = set(["GEMINI", "NIRI", "IMAGE"])

Public methods::

  Not Implemented
