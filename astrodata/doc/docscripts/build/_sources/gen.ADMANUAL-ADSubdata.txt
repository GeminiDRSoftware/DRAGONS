


Using Slices and "Subdata"
--------------------------

.. image:: diagrams/sharedHDUs.*
   :scale: 30%
   :align: center

AstroData instances are presented as lists of AstroData instances.
However, internally the list is merely a list of extensions and the
*AstroData.getitem(..)* function (which implements the "[]" syntax)
creates AstroData instances on the fly when called. Such instances
share information in memory with their parent instance. This is in
line with the general operation of pyfits and numpy, and in general
how Python handles objects. This allows efficient use of memory and
disk I/O. To make copies one must explicitly ask for copies. Thus when
one takes a slice of a numpy array, that slice, although possibly of a
different dimensionality and certainly of range, is really just a view
onto the original memory, changes to the slice affect the original. If
one takes a subset of an AstroData instance's HDUList, then the save
HDUs are present in both the original and the sub-data. To make a
separate copy one must use the *deepcopy* built-in function (see
below).

As the diagram indicates, when taking a subset of data from an
AstroData instance using the square brackets operator, you receive a
newly created AstroData instance which is associated only with those
HDUs identified. Changes to a shared HDU's data or header member will
be reflected in both AstroData instances. Generally speaking this is
what you want for efficient operation. If you do want to have entirely
separate data, such that changes to the data sections of one do not
affect the other, use the python deepcopy operator:

.. code-block:: python
    :linenos:

    
    from copy import deepcopy
    
    ad = AstroData("dataset.fits")
    scicopy = deepcopy(ad["SCI"])


If on the other hand all you want is to avoid changing the original
dataset on disk, and do not need the original data, untransformed, in
memory along with the transformed version, which is the usual case,
then you can write the AstroData subdata instance to a new filename:

.. code-block:: python
    :linenos:

    
    from astrodata import AstroData
    
    ad = AstroData("dataset.fits")
    scicopy = ad["SCI"]
    scicopy.write("datasetSCI.fits")





