.. supptools:

Supplemental
************

.. _auto_mos:

Using reduce to tile and  mosaic datasets
=========================================

The DRAGONS ``reduce`` script provides mechanisms by which users can generate
tiled or mosaicked images from data directly on the command line.

Recall, when using the ``reduce`` option [-r, --recipe], ``reduce`` will find a
specified primitive name if it is passed with this option. In the DRAGONS
*geminidr* package, mosaic and tiling functionality are provided by the primitives
``mosaicDetectors`` and ``tileArrays``. These will perform mosaic and tiling
operations on both GMOS and GSAOI data.

.. note:: The dataset must be *prepared*.

Example
-------

1) Run mosaic on a full dataset, i.e., SCI, VAR, DQ arrays, plus any OBJMASK
   and reference catalog.

::

 $ reduce -r mosaicDetectors N20170913S0209_prepared.fits

	--- reduce, vv2.0.8 ---
 All submitted files appear valid
 Found 'mosaicDetectors' as a primitive.
 =============================================================
 RECIPE: mosaicDetectors
 =============================================================
    PRIMITIVE: mosaicDetectors
    --------------------------
         MosaicAD Working on GMOS IMAGE
         Building mosaic, converting data ...
    MosaicAD working on data arrays ...
    Working on VAR arrays ...
    Working on DQ arrays ...
    Working on OBJMASK arrays ...
    Keeping REFCAT ...
    Updated filename: N20170913S0209_mosaic.fits
    .
 Wrote N20170913S0209_mosaic.fits in output directory

 reduce completed successfully.


2) Run tiling on a full dataset, i.e., SCI, VAR, DQ arrays, plus any OBJMASK
   and reference catalog.

::

 $ reduce -r tileArrays N20170913S0209_prepared.fits

          --- reduce, vv2.0.8 ---
 All submitted files appear valid
 Found 'tileArrays' as a primitive.
 ===========================================================
 RECIPE: tileArrays
 ===========================================================
   PRIMITIVE: tileArrays
   ---------------------
   Tile arrays parameter, tile_all is False
      PRIMITIVE: mosaicDetectors
      --------------------------
         MosaicAD Working on GMOS IMAGE
      	 Building mosaic, converting data ...
      MosaicAD working on data arrays ...
      Working on VAR arrays ...
      Working on DQ arrays ...
      Working on OBJMASK arrays ...
      Tiling OBJCATS ...
      Keeping REFCAT ...
      Updated filename: N20170913S0209_tiled.fits
   .
 .
 Wrote N20170913S0209_tiled.fits in output directory

 reduce completed successfully.

You can request that tiling be done onto a single output grid (one extension) with
the ``tile_all`` parameter, which can be passed on the command line as well.

::

 $ reduce -p tile_all=True -r tileArrays N20170913S0209_prepared.fits

          --- reduce, vv2.0.8 ---
 All submitted files appear valid
 Found 'tileArrays' as a primitive.
 ===========================================================
 RECIPE: tileArrays
 ===========================================================
   PRIMITIVE: tileArrays
   ---------------------
   Tile arrays parameter, tile_all is True
      PRIMITIVE: mosaicDetectors
      --------------------------
         MosaicAD Working on GMOS IMAGE
      	 Building mosaic, converting data ...
      MosaicAD working on data arrays ...
      Working on VAR arrays ...
      Working on DQ arrays ...
      Working on OBJMASK arrays ...
      Tiling OBJCATS ...
      Keeping REFCAT ...
      Updated filename: N20170913S0209_tiled.fits
   .
 .
 Wrote N20170913S0209_tiled.fits in output directory

 reduce completed successfully.
