.. supptools:

Supplemental
************

.. _auto_mos:

reduce and mosaic
=================

The DRAGONS ``reduce`` script provides a mechanism by which users can generate
mosaicked images from data directly on the command line. Recall, when using
the ``reduce`` option [-r, --recipe], ``reduce`` will find a specified primitive
name if it is passed with this option. In the DRAGONS *geminidr* package, mosaic
functionality is provided by the primitive ``mosaicDetectors`` and will perform
mosaic operations on both GMOS and GSAOI data.

.. note:: The dataset must be *prepared*. 

Example
-------

1) Run mosaic on a full dataset, i.e., SCI, VAR, DQ arrays, plus any OBJMASK
   and reference catalog.

::

 $ reduce -r mosaicDetectors N20170913S0209_prepared.fits
   
	--- reduce, v2.0 (beta) ---
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

 
