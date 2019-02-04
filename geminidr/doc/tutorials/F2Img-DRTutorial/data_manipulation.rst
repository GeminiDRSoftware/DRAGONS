
.. |astrodata| replace:: :mod:`~astrodata`
.. |gemini_instruments| replace:: :mod:`gemini_instruments`

.. _`DRAGONS`: https://github.com/GeminiDRSoftware/DRAGONS


.. _loading_f2_data:

Data Manipulation
-----------------

Here we will show you how to load and manipulate a Flamingos 2 FITS file using
DRAGONS' |astrodata| and |gemini_instruments| packages.

For this case, we will be using an arbitrary file which can be `downloaded
directly from the GEMINI archive here <https://archive.gemini.edu/file/S20130622S0040.fits>`_.

First, let's start by opening Python and importing |astrodata|:

.. ipython::

   In [1]: import astrodata

Nothing should be printed. If you get an error message (usually long and messy),
please contact us for further support. Now, let us load
`an arbitrary F2 data file <https://archive.gemini.edu/file/S20130622S0040.fits>`_
using :func:`astrodata.open`.

.. ipython::

   In [2]: ad = astrodata.open('S20130622S0040.fits')

Again, nothing should be printed. You have now a generic astrodata object which
is stored inside ``ad`` variable:

.. ipython::

   In [3]: ad

This does not tells us much. So let's try to get a bit more information using
:func:`ad.info()`:

.. ipython::

   In [4]: ad.info()

If you are running within IPython or a Jupyter notebook, you can type ``ad.`` and
press ``TAB``. With that you will see all the available object's attributes
available, like ``ad.telescope()`` and ``ad.instrument()``:

.. ipython::

   In [5]: ad.telescope()

   In [6]: ad.instrument()

``ad`` is an object that represents a FITS file with a single extension. It knows
that it is a data from the GEMINI South telescope and that it was obtained with
F2 but there is no much more information than that. For more details,
|astrodata| requires the :class:`~astrodata.AstroData` class to be subclassed for
each instrument. This subclass contains the tags and parses the metadata as
attributes.

Luckly, F2 already has its own class based on :class:`~astrodata.AstroData` and
this class comes with the `DRAGONS`_ meta-package. To use it, you have to simply
import |gemini_instruments|, as shown below:

.. ipython::

   In [7]: import gemini_instruments

Now, let us open the same file and store into a new variable called ``f2_ad``
simply because we may want to compare these two variables at some point.

.. ipython::

   In [8]: f2_ad = astrodata.open('S20130622S0040.fits')

Note that this object has a different class:

.. ipython::

   In [9]: f2_ad

As you can see above, instead of having an :class:`astrodata.fits.AstroDataFits`
object, we have a :class:`gemini_instruments.f2.adclass.AstroDataF2` object. So,
yes, there is a lot of things happening behind the curtains. Our FITS file was
loaded and its meta-data was used by |astrodata| and |gemini_instruments|
together to build a :class:`~gemini_instruments.f2.adclass.AstroDataF2` object,
which knows much more about itself:

.. ipython::

   In [10]: f2_ad.info()

Note the ``TAGS`` written in the top of the printed message. It gives us several
hints about this file, e.g., it is a calibration (CAL) file, a FLAT file
obtained in IMAGE mode with the LAMPON and this file was not processed (RAW).
These ``TAGS`` should be updated as we walk through the data reduction steps and
is used by `DRAGONS`_ to select which object is processed by which function.

.. ipython::

   In [11]: f2_ad.telescope()

   In [12]: f2_ad.instrument()

   In [13]: f2_ad.airmass()

   In [14]: f2_ad.filter_name()

We can compare how many attributes ``ad`` and ``f2_ad`` have by using the
built-in :func:`dir` and :func:`len` functions:

.. ipython::

   In [15]: len(dir(ad))

   In [16]: len(dir(f2_ad))

Within DRAGONS context, these new attributes are called **descriptors**. You can
find all the descriptors that |astrodata| could find in a
:class:`~gemini_instruments.f2.AstroDataF2` object via
:attr:`~gemini_instruments.f2.AstroDataF2.descriptors` attribute:

.. ipython::

   In [17]: f2_ad.descriptors[:5]

The ``[:5]`` was appended to limit the (very large) output to make this tutorial
cleaner. These descriptors are the `DRAGONS`_ way to access the meta-data in an
uniform fashion. This allows us to re-use methods without having to worry about
how to access the meta-data and is particularly useful when building
data-reduction pipelines. But, for now, let's continue exploring our data.

The :meth:`~astrodata.AstroData.info()` method told us before that this object
has a single extension which can be access using Python standard indexing. This
extension can be used to access the data array and the header of the FITS file:

.. ipython::

   In [18]: f2_ad[0].data

   In [19]: f2_ad[0].hdr[:10]

Again, the ``[:10]`` was use just to make this document cleaner.

The :attr:`~astrodata.AstroData.data` and the
:attr:`~astrodata.AstroData.header` attributes makes it easier to manipulate the
data directly.

Now, if we want to actually display our data, we have almost all the resources
to do so. We simply need to import :mod:`matplotlib` and display the image as we
would do for a NumPy array:

.. ipython::

   In [19]: import matplotlib.pyplot as plt

   In [20]: data_array = f2_ad[0].data

   In [21]: vmin = data_array[0].mean() - data_array[0].std()

   In [22]: vmax = data_array[0].mean() + data_array[0].std()

   In [23]: plt.imshow(data_array[0], vmin=vmin, vmax=vmax)

   In [24]: plt.savefig("my_figure.png")

.. figure:: ../my_figure.png
   :align: center

   ./data/my_figure.png


Common operations (like add, subtract, divide, and multiply) can be performed
directly on the :class:`~gemini_instruments.f2.AstroDataF2` object:

.. ipython::

   In [25]: print('Array mean before multiplication: ', f2_ad[0].data.mean())

   In [26]: new_f2_ad = f2_ad * 2

   In [27]: print('Array mean after multiplication by two: ', new_f2_ad[0].data.mean())

Before we write down this new object to the disk as a new FITS file, we need to
update the filename it should have:

.. ipython::

   In [28]: foo = f2_ad.info()

.. todo::

    What about getting/setting filenames?

.. todo::

    What's next?