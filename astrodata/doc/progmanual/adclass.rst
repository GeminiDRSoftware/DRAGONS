.. astrodata.rst

.. _astrodata:

*************************
AstroData and Derivatives
*************************

.. todo:: remove this, for jenkins tests

The |AstroData| class is the main interface to the package. When opening files
or creating new objects, a derivative of this class is returned, as the
|AstroData| class is not intended to be used directly. It provides the logic to
calculate the :ref:`tag set <ad_tags>` for an image, which is common to all
data products. Aside from that, it lacks any kind of specialized knowledge
about the different instruments that produce the FITS files. More importantly,
it defines two methods (``info`` and ``load``) as abstract, meaning that the
class cannot be instantiated directly: a derivative must implement those
methods in order to be useful. Such derivatives can also implement descriptors,
which provide processed metadata in a way that abstracts the user from the raw
information (e.g., the keywords in FITS headers).

|AstroData| does define a common interface, though. Much of it consists on
implementing semantic behavior (access to components through indices, like a
list; arithmetic using standard operators; etc), mostly by implementing
standard Python methods:

* Defines a common ``__init__`` function.

* Implements ``__deepcopy__``.

* Implements ``__iter__`` to allow sequential iteration over the main set of
  components (e.g., FITS science HDUs).

* Implements ``__getitem__`` to allow data slicing (e.g., ``ad[2:4]`` returns
  a new |AstroData| instance that contains only the third and fourth main
  components).

* Implements ``__delitem__`` to allow for data removal based on index. It does
  not define ``__setitem__``, though. The basic AstroData series of classes
  only allows to append new data blocks, not to replace them in one sweeping
  move.

* Implements ``__iadd__``, ``__isub__``, ``__imul__``, ``__itruediv__``, and
  their not-in-place versions, based on them.

There are a few other methods. For a detailed discussion, please refer to the
:ref:`api`.

.. _tags_prop_entry:

The ``tags`` Property
=====================

Additionally, and crucial to the package, AstroData offers a ``tags`` property,
that under the hood calculates textual tags that describe the object
represented by an instance, and returns a set of strings. Returning a set (as
opposed to a list, or other similar structure) is intentional, because it is
fast to compare sets, e.g., testing for membership; or calculating intersection,
etc., to figure out if a certain dataset belongs to an arbitrary category.

The implementation for the tags property is just a call to
``AstroData._process_tags()``. This function implements the actual logic behind
calculating the tag set (described :ref:`below <ad_tags>`). A derivative class
could redefine the algorithm, or build upon it.


Writing an ``AstroData`` Derivative
===================================

The first step when creating new |AstroData| derivative hierarchy would be to
create a new class that knows how to deal with some kind of specific data in a
broad sense.

|AstroData| implements both ``.info()`` and ``.load()`` in ways that are
specific to FITS files. It also introduces a number of FITS-specific methods
and properties, e.g.:

* The properties ``phu`` and ``hdr``, which return the primary header and
  a list of headers for the science HDUs, respectively.

* A ``write`` method, which will write the data back to a FITS file.

* A ``_matches_data`` **static** method, which is very important, involved in
  guiding for the automatic class choice algorithm during data loading. We'll
  talk more about this when dealing with :ref:`registering our classes
  <class_registration>`.

It also defines the first few descriptors, which are common to all Gemini data:
``instrument``, ``object``, and ``telescope``, which are good examples of simple
descriptors that just map a PHU keyword without applying any conversion.

A typical AstroData programmer will extend this class (|AstroData|). Any of
the classes under the ``gemini_instruments`` package can be used as examples,
but we'll describe the important bits here.


Create a package for it
-----------------------

This is not strictly necessary, but simplifies many things, as we'll see when
talking about *registration*. The package layout is up to the designer, so you
can decide how to do it. For DRAGONS we've settled on the following
recommendation for our internal process (just to keep things familiar)::

    gemini_instruments
        __init__.py
        instrument_name
            __init__.py
            adclass.py
            lookup.py

Where ``instrument_name`` would be the package name (for Gemini we group all
our derivative packages under ``gemini_instruments``, and we would import
``gemini_instruments.gmos``, for example). ``__init__.py`` and ``adclass.py``
would be the only required modules under our recommended layout, with
``lookup.py`` being there just to hold hard-coded values in a module separate
from the main logic.

``adclass.py`` would contain the declaration of the derivative class, and
``__init__.py`` will contain any code needed to register our class with the
|AstroData| system upon import.


Create your derivative class
----------------------------

This is an excerpt of a typical derivative module::

    from astrodata import astro_data_tag, astro_data_descriptor, TagSet
    from astrodata import AstroData

    from . import lookup

    class AstroDataInstrument(AstroData):
        __keyword_dict = dict(
            array_name = 'AMPNAME',
            array_section = 'CCDSECT'
        )

        @staticmethod
        def _matches_data(source):
            return source[0].header.get('INSTRUME', '').upper() == 'MYINSTRUMENT'

        @astro_data_tag
        def _tag_instrument(self):
           return TagSet(['MYINSTRUMENT'])

        @astro_data_tag
        def _tag_image(self):
            if self.phu.get('GRATING') == 'MIRROR':
                return TagSet(['IMAGE'])

        @astro_data_tag
        def _tag_dark(self):
            if self.phu.get('OBSTYPE') == 'DARK':
                return TagSet(['DARK'], blocks=['IMAGE', 'SPECT'])

        @astro_data_descriptor
        def array_name(self):
            return self.phu.get(self._keyword_for('array_name'))

        @astro_data_descriptor
        def amp_read_area(self):
            ampname = self.array_name()
            detector_section = self.detector_section()
            return "'{}':{}".format(ampname, detector_section)

.. note::
   An actual Gemini Facility Instrument class will derive from
   ``gemini_instruments.AstroDataGemini``, but this is irrelevant
   for the example.

The class typically relies on functionality declared elsewhere, in some
ancestor, e.g., the tag set computation and the ``_keyword_for`` method are
defined at |AstroData|.

Some highlights:

* ``__keyword_dict``\ [#keywdict]_ defines one-to-one mappings, assigning a more
  readable moniker for an HDU header keyword. The idea here is to prevent
  hard-coding the names of the keywords, in the actual code. While these are
  typically quite stable and not prone to change, it's better to be safe than
  sorry, and this can come in useful during instrument development, which is
  the more likely source of instability. The actual value can be extracted by
  calling ``self._keyword_for('moniker')``.

* ``_matches_data`` is a static method. It does not have any knowledge about
  the class itself, and it does not work on an *instance* of the class: it's
  a member of the class just to make it easier for the AstroData registry to
  find it. This method is passed some object containing cues of the internal
  structure and contents of the data. This could be, for example, an instance
  of ``HDUList``. Using these data, ``_matches_data`` must return a boolean,
  with ``True`` meaning "I know how to handle this data".

  Note that ``True`` **does not mean "I have full knowledge of the data"**. It
  is acceptable for more than one class to claim compatibility. For a GMOS FITS
  file, the classes that will return ``True`` are: |AstroData| (because it is
  a FITS file that comply with certain minimum requirements),
  `~gemini_instruments.gemini.AstroDataGemini` (the data contains Gemini
  Facility common metadata), and `~gemini_instruments.gmos.AstroDataGmos` (the
  actual handler!).

  But this does not mean that multiple classes can be valid "final" candidates.
  If AstroData's automatic class discovery finds more than one class claiming
  matching with the data, it will start discarding them on the basis of
  inheritance: any class that appears in the inheritance tree of another one is
  dropped, because the more specialized one is preferred. If at some point the
  algorithm cannot find more classes to drop, and there is more than one left
  in the list, an exception will occur, as AstroData will have no way to choose
  one over the other.

* A number of "tag methods" have been declared. Their naming is a convention,
  at the end of the day (the "``_tag_``" prefix, and the related "``_status_``"
  one, are *just hints* for the programmer): each team should establish
  a convention that works for them. What is important here is to **decorate**
  them using `~astrodata.astro_data_tag`, which earmarks the method so that it
  can be discovered later, and ensures that it returns an appropriate value.

  A tag method will return either a `~astrodata.TagSet` instance (which can be
  empty), or ``None``, which is the same as returning an empty
  `~astrodata.TagSet`\ [#tagset1]_.

  **All** these methods will be executed when looking up for tags, and it's up
  to the tag set construction algorithm (see :ref:`ad_tags`) to figure out the final
  result.  In theory, one **could** provide *just one* big method, but this is
  feasible only when the logic behind deciding the tag set is simple. The
  moment that there are a few competing alternatives, with some conditions
  precluding other branches, one may end up with a rather complicated dozens of
  lines of logic. Let the algorithm do the heavy work for you: split the tags
  as needed to keep things simple, with an easy to understand logic.

  Also, keeping the individual (or related) tags in separate methods lets you
  exploit the inheritance, keeping common ones at a higher level, and
  redefining them as needed later on, at derived classes.

  Please, refer to `~gemini_instruments.gemini.AstroDataGemini`,
  `~gemini_instruments.gmos.AstroDataGmos`, and
  `~gemini_instruments.gnirs.AstroDataGnirs` for examples using most of the
  features.

* The `astrodata.AstroData.read` method calls the `astrodata.fits.read_fits`
  function, which uses metadata in the FITS headers to determine how the data
  should be stored in the |AstroData| object. In particular, the ``EXTNAME``
  and ``EXTVER`` keywords are used to assign individual FITS HDUs, using the
  same names (``SCI``, ``DQ``, and ``VAR``) as Gemini-IRAF for the ``data``,
  ``mask``, and ``variance`` planes.  A ``SCI`` HDU *must* exist if there is
  another HDU with the same ``EXTVER``, or else an error will occur.

  If the raw data do not conform to this format, the `astrodata.AstroData.read`
  method can be overridden by your class, by having it call the
  `astrodata.fits.read_fits` function with an additional parameter,
  ``extname_parser``, that provides a function to modify the header. This
  function will be called on each HDU before further processing. As an example,
  the SOAR Adaptive Module Imager (SAMI) instrument writes raw data as
  a 4-extension MEF file, with the extensions having ``EXTNAME`` values
  ``im1``, ``im2``, etc. These need to be modified to ``SCI``, and an
  appropriate ``EXTVER`` keyword added` [#extver]_\. This can be done by
  writing a suitable ``read`` method for the ``AstroDataSami`` class::

    @classmethod
    def read(cls, source, extname_parser=None):
        def sami_parser(hdu):
            m = re.match('im(\d)', hdu.header.get('EXTNAME', ''))
            if m:
                hdu.header['EXTNAME'] = ('SCI', 'Added by AstroData')
                hdu.header['EXTVER'] = (int(m.group(1)), 'Added by AstroData')

        return super().read(source, extname_parser=extname_parser)


* *Descriptors* will make the bulk of the class: again, the name is arbitrary,
  and it should be descriptive. What *may* be important here is to use
  `~astrodata.astro_data_descriptor` to decorate them. This is *not required*,
  because unlike tag methods, descriptors are meant to be called explicitly by
  the programmer, but they can still be marked (using this decorator) to be
  listed when calling the ``descriptors`` property. The decorator does not
  alter the descriptor input or output in any way, so it is always safe to use
  it, and you probably should, unless there's a good reason against it (e.g.,
  if a descriptor is deprecated and you don't want it to show up in lookups).

  More detailed information can be found in :ref:`ad_descriptors`.


.. _class_registration:

Register your class
-------------------

Finally, you need to include your class in the **AstroData Registry**. This is
an internal structure with a list of all the |AstroData|\-derived classes that
we want to make available for our programs. Including the classes in this
registry is an important step, because a file should be opened using
`astrodata.open` or `astrodata.create`, which uses the registry to identify
the appropriate class (via the ``_matches_data`` methods), instead of having
the user specify it explicitly.

The version of AstroData prior to DRAGONS had an auto-discovery mechanism, that
explored the source tree looking for the relevant classes and other related
information. This forced a fixed directory structure (because the code needed
to know where to look for files), and gave the names of files and classes
semantic meaning (to know *which* files to look into, for example). Aside from
the rigidness of the scheme, this introduced all sort of inefficiencies,
including an unacceptably high overhead when importing the AstroData package
for the first time during execution.

In this new version of AstroData we've introduced a more manageable scheme,
that places the discovery responsibility on the programmer. A typical
``__init__.py`` file on an instrument package will look like this::

    __all__ = ['AstroDataMyInstrument']

    from astrodata import factory
    from .adclass import AstroDataMyInstrument

    factory.addClass(AstroDataMyInstrument)

The call to ``factory.addClass`` is the one registering the class. This step
**needs** to be done **before** the class can be used effectively in the
AstroData system. Placing the registration step in the ``__init__.py`` file is
convenient, because importing the package will be enough!

Thus, a script making use of DRAGONS' AstroData to manipulate GMOS data
could start like this::

    import astrodata
    from gemini_instruments import gmos

    ...

    ad = astrodata.open(some_file)

The first import line is not needed, technically, because the ``gmos`` package
will import it too, anyway, but we'll probably need the ``astrodata`` package
in the namespace anyway, and it's always better to be explicit. Our
typical DRAGONS scripts and modules start like this, instead::

    import astrodata
    import gemini_instruments

``gemini_instruments`` imports all the packages under it, making knowledge
about all Gemini instruments available for the script, which is perfect for a
multi-instrument pipeline, for example. Loading all the instrument classes is
not typically a burden on memory, though, so it's easier for everyone to take
the more general approach. It also makes things easier on the end user, because
they won't need to know internal details of our packages (like their naming
scheme). We suggest this "*cascade import*" scheme for all new source trees,
letting the user decide which level of detail they need.

As an additional step, the ``__init__.py`` file in a package may do extra
initialization. For example, for the Gemini modules, one piece of functionality
that is shared across instruments is a descriptor that translates a filter's
name (say "u" or "FeII") to its central wavelength (e.g.,
0.35µm, 1.644µm). As it is a rather common function for us, it is implemented
by `~gemini_instruments.gemini.AstroDataGemini`. This class **does not know**
about its daughter classes, though, meaning that it **cannot know** about the
filters offered by their instruments. Instead, we offer a function that can
be used to update the filter → wavelength mapping in
`gemini_instruments.gemini.lookup` so that it is accessible by the
`~gemini_instruments.gemini.AstroDataGemini`\-level descriptor. So our
``gmos/__init__.py`` looks like this::

    __all__ = ['AstroDataGmos']

    from astrodata import factory
    from ..gemini import addInstrumentFilterWavelengths
    from .adclass import AstroDataGmos
    from .lookup import filter_wavelengths

    factory.addClass(AstroDataGmos)
    # Use the generic GMOS name for both GMOS-N and GMOS-S
    addInstrumentFilterWavelengths('GMOS', filter_wavelengths)

where `~gemini_instruments.gemini.addInstrumentFilterWavelengths` is provided
by the ``gemini`` package to perform the update in a controlled way.

We encourage package maintainers and creators to follow such explicit
initialization methods, driven by the modules that add functionality
themselves, as opposed to active discovery methods on the core code. This
favors decoupling between modules, which is generally a good idea.

.. rubric:: Footnotes

.. [#keywdict] Note that the keyword dictionary is a "private" property of the
   class (due to the double-underscore prefix). Each class can define its own
   set, which will not be replaced by derivative classes. ``_keyword_for`` is
   aware of this and will look up each class up the inheritance chain, in turn,
   when looking up for keywords.

.. [#tagset1] Notice that the example functions will return only
   a `~astrodata.TagSet`, if appropriate. This is OK, remember that *every
   function* in Python returns a value, which will be ``None``, implicitly, if
   you don't specify otherwise.

.. [#extver] An ``EXTVER`` keyword is not formally required as the
   `astrodata.fits.read_fits` method will assign the lowest available integer
   to a ``SCI`` header with no ``EXTVER`` keyword (or if its value is -1). But
   we wish to be able to identify the original ``im1`` header by assigning it
   an ``EXTVER`` of 1, etc.
