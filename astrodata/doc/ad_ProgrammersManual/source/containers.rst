.. containers.rst

.. _containers:

***************
Data Containers
***************

A third, and very important part of the AstroData core package is the data
container. We have chosen to extend Astropy's NDData\ [#nddata]_ with our own
requirements, particularly lazy-loading of data using by opening the FITS files
in read-only, memory-mapping mode, and exploiting the windowing capability of
PyFITS\ [#pyfits]_ (using ``section``) to reduce our memory requirements, which
becomes important when reducing data (e.g., stacking).

We document our container for completeness and for reference, but note that its
use is associated to ``FitsProvider``. If you're implementing an alternative
data provider, you do not need to follow our design.

We'll describe here how we depart from NDData, and how do we integrate the data
containers with the rest of the package. Please, refer to NDData for the full
interface.

Our main data container is ``astrodata.nddata.NDAstroData``. Fundamentally, it
is a derivative of ``astropy.nddata.NDData``, plus a number of mixins to add
functionality::

    class NDAstroData(NDArithmeticMixin, NDSlicingMixin, NDData):
        ...

This allows us out of the box to have proper arithmetic with error
propagation, and slicing the data with the array syntax.

Our first customization is ``NDAstroData.__init__``. It relies mostly on the
upstream initialization, but customizes it because our class is initialized
with lazy-loaded data wrapped around a custom class
(``astrodata.fits.FitsLazyLoadable``) that mimics a PyFITS HDU instance just
enough to play along with NDData's initialization code.

``FitsLazyLoadable`` is an integral part of our memory-mapping scheme, and
among other things it will scale data on the fly, as memory-mapped FITS data
can only be read unscaled. Our NDAstroData redefines the properties ``data``,
``uncertainty``, and ``mask``, in two ways:

* To deal with the fact that our class is storing ``FitsLazyLoadable``
  instances, not arrays, as ``NDData`` would expect. This is to keep data out
  of memory as long as possible.

* To replace lazy-loaded data with a real in-memory array, under certain
  conditions (e.g., if the data is modified, as we won't apply the changes to the
  original file!)

Our obsession with lazy-loading and discarding data is directed to reduce
memory fragmentation as much as possible. This is a real problem that can hit
applications dealing with large arrays, particularly when using Python. Given
the choice to optimize for speed or for memory consumption, we've chosen the
latter, which is the more pressing issue.

Another addition of as is the ``variance`` property as a convenience for the
user.. Astropy, so far, only provides a standard deviation class for storing
uncertainties and the code to propagate errors stored this way already
exists. However, our coding elsewhere is greatly simplified if we are able
to access and set the variance directly.

At last, we've added another new property, ``window``, that can be used to
explicitly exploit the PyFITS ``section`` property, to (again) avoid loading
unneeded data to memory. This property returns an instance of ``NDWindowing``
which, when sliced, in turn produces an instance of ``NDWindowingAstroData``,
itself a proxy of ``NDAstroData``. This scheme may seem complex, but it was
deemed the easiest and cleanest way to achieve the result that we were looking
for.

.. note::

   We expect to make changes to ``NDAstroData`` in future releases. In particular,
   we plan to make use of the ``wcs`` and ``unit`` attributes provided by the
   ``NDData`` class and increase the use of memory-mapping by default. These
   changes mostly represent increased functionality and we anticipate a high
   (and possibly full) degree of backward compatibility.

.. rubric:: Footnotes

.. [#nddata] Astropy: `N-dimensional datasets <http://docs.astropy.org/en/stable/nddata>`_

.. [#pyfits] I mention PyFITS because of familiarity and for short, but in reality
   we're using Astropy's ``fits.io`` module.
