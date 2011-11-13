


Single HDU AstroData Instances Overview
---------------------------------------

AstroData instances are presented as iterable containera of AstroData
instances (or sub-data, see ref:"ADREF_subdata"). Thus in the
following example:

.. code-block:: python
    :linenos:

    
    from astrodata.data import AstroData
    
    ad = AstroData("dataset.fits")
    
    for ext in ad:
        print ext.gain()


The variable "ad" holds an AstroData instance, of course, and so does
the "ext" instance. The ext variable, furthermore, is guaranteed to
have just one HDU in its collection. Some functions are designed to
work only on Single-HDU AstroData instances, such as the "data"
member. For a multiple-HDU AstroData instance, the data member is
ambiguous, each HDU has a "data" member (the numpy array).

