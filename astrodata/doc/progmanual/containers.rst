.. containers.rst

.. _containers:

***************
Data Containers
***************

A third, and very important part of the AstroData core package is the data
container. We have chosen to extend Astropy's |NDData| with our own
requirements, particularly lazy-loading of data using by opening the FITS files
in read-only, memory-mapping mode, and exploiting the windowing capability of
`astropy.io.fits` (using ``section``) to reduce our memory requirements, which
becomes important when reducing data (e.g., stacking).

We'll describe here how we depart from |NDData|, and how do we integrate the
data containers with the rest of the package. Please refer to |NDData| for the
full interface.

Our main data container is `astrodata.NDAstroData`. Fundamentally, it is
a derivative of `astropy.nddata.NDData`, plus a number of mixins to add
functionality::

    class NDAstroData(AstroDataMixin, NDArithmeticMixin, NDSlicingMixin, NDData):
        ...

This allows us out of the box to have proper arithmetic with error
propagation, and slicing the data with the array syntax.

Our first customization is ``NDAstroData.__init__``. It relies mostly on the
upstream initialization, but customizes it because our class is initialized
with lazy-loaded data wrapped around a custom class
(`astrodata.fits.FitsLazyLoadable`) that mimics a `astropy.io.fits` HDU
instance just enough to play along with |NDData|'s initialization code.

``FitsLazyLoadable`` is an integral part of our memory-mapping scheme, and
among other things it will scale data on the fly, as memory-mapped FITS data
can only be read unscaled. Our NDAstroData redefines the properties ``data``,
``uncertainty``, and ``mask``, in two ways:

* To deal with the fact that our class is storing ``FitsLazyLoadable``
  instances, not arrays, as |NDData| would expect. This is to keep data out
  of memory as long as possible.

* To replace lazy-loaded data with a real in-memory array, under certain
  conditions (e.g., if the data is modified, as we won't apply the changes to the
  original file!)

Our obsession with lazy-loading and discarding data is directed to reduce
memory fragmentation as much as possible. This is a real problem that can hit
applications dealing with large arrays, particularly when using Python. Given
the choice to optimize for speed or for memory consumption, we've chosen the
latter, which is the more pressing issue.

We've added another new property, ``window``, that can be used to
explicitly exploit the `astropy.io.fits`'s ``section`` property, to (again)
avoid loading unneeded data to memory. This property returns an instance of
``NDWindowing`` which, when sliced, in turn produces an instance of
``NDWindowingAstroData``, itself a proxy of ``NDAstroData``. This scheme may
seem complex, but it was deemed the easiest and cleanest way to achieve the
result that we were looking for.

The base ``NDAstroData`` class provides the memory-mapping functionality,
with other important behaviors added by the ``AstroDataMixin``, which can
be used with other |NDData|-like classes (such as ``Spectrum1D``) to add
additional convenience.

One addition is the ``variance`` property, which allows direct access and
setting of the data's uncertainty, without the user needing to explicitly wrap
it as an ``NDUncertainty`` object. Internally, the variance is stored as an
``ADVarianceUncertainty`` object, which is subclassed from Astropy's standard
``VarianceUncertainty`` class with the addition of a check for negative values
whenever the array is accessed.

``NDAstroDataMixin`` also changes the default method of combining the ``mask``
attributes during arithmetic operations from ``logical_or`` to ``bitwise_or``,
since the individual bits in the mask have separate meanings.

The way slicing affects the ``wcs`` is also changed since DRAGONS regularly
uses the callable nature of ``gWCS`` objects and this is broken by the standard
slicing method.

Finally, the additional image planes and tables stored in the ``meta`` dict
are exposed as attributes of the ``NDAstroData`` object, and any image planes
that have the same shape as the parent ``NDAstroData`` object will be handled
by ``NDWindowingAstroData``. Sections will be ignored when accessing image
planes with a different shape, as well as tables.


.. note::

   We expect to make changes to ``NDAstroData`` in future releases. In particular,
   we plan to make use of the ``unit`` attribute provided by the
   |NDData| class and increase the use of memory-mapping by default. These
   changes mostly represent increased functionality and we anticipate a high
   (and possibly full) degree of backward compatibility.
