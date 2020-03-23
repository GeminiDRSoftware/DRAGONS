.. providers.rst

.. _providers:

**************
Data Providers
**************

``AstroData`` derivative classes act as a front end. Most of the heavy lifting
is actually performed by a ``DataProvider`` class. There will typically be one
data provider per kind of data structure (so far, DRAGONS offers only
``astrodata.fits.FitsProvider``), and possibly one data provider **proxy**,
used to simplify the handling of data slicing (mapping most of it operations to
a regular data provider.)

The data provider acts as a hierarchical data storage. At the **top level**, it
contains:

* A sequence of "extensions", representing individual data planes and their
  associated metadata, likely to
  represent separate detectors or amplifiers. One can
  access these extensions by index (e.g., ``ad[5]``). Indexing starts at 0,
  following Python's convention.
* Global objects like masks or tables affecting all the extensions.

Each extension, in turn, is an instance of a Data Container, keeping important
metadata (e.g., a FITS HDU's header) and the main data for the extension (e.g., the
data for a SCI extension, on Gemini data), along with any other associated data
(masks, variance plane, tables, etc).

``astrodata.core.DataProvider`` is, again, an abstract class, defining the
minimum interface expected from a data provider. This interface is described in
greater detail in the :ref:`api_refguide`, but among other things, one would
need to implement:

* ``is_settable``: AstroData exposes attributes from its data provider through
  its own ``__getattr__`` and ``__setattr__``. When trying to set a value for
  an attribute, ``AstroData`` will use this method to discover whether the
  attribute can be modified.
* ``append``: a very important method, used to add new top-level components to
  the provider.
* ``__getitem__``: which returns a sliced **view**\ [#viewnote]_ of the
  provider itself, meant to work with isolated extensions (examples of such a
  view are instances of ``astrodata.fits.FitsProviderProxy``). The view should
  behave in almost every way as a normal provider.
* ``__len__``: number of science extensions contained in this instance.
* ``__iadd__``, ``__isub__``, ``__imul__``, ``__itruediv__``; used to perform
  in-place operations over the data.
* ``data``, ``uncertainty``, ``mask``, ``variance``: properties used to access
  certain common content. These methods generally return lists, with one
  element per extension.

There are also a number of properties that are not declared as abstract, but
still need to be reimplemented if one would want any kind of proper behavior
from the class: ``exposed`` (used to determine if a certain attribute is to be
"exposed" to the user through the AstroData class), ``is_sliced``, and
``is_single``. Of particular interest is this later one: ``is_single`` is a
predicate that should return ``True`` only if a data provider has been sliced
using a single index, e.g.::

    >>> d1 = provider[:4]
    >>> d1.is_sliced, d1.is_single
    (True, False)
    >>> d2 = provider[3]
    >>> d2.is_sliced, d2.is_single
    (True, True)

This is important for the AstroData interface. When a data provider is being
considered a "single" slice, the behavior of many methods change. For example,
we mentioned that the ``data`` property *generally* returns a list. **If the
data provider in question is a single slice, then data would return a single
(i.e., scalar) element**. This behavior is often seen also in :ref:`ad_descriptors`.
Refer always to the to documentation of a method to figure out how they behave. As
programmers, you should always include this explicitly in the documentation,
even if it's implicit to AstroData.

Implementation Guidelines
=========================

AstroData does not impose any restriction on how to organize the data
internally, or how to deal with slicing. On slicing, we chose to use a "proxy"
class for ``FitsProvider``. So, when sliced (through __getitem__), a
``FitsProvider`` will return a ``FitsProviderProxy``, which is also a
descendant of ``DataProvider`` and reproduces the interface of its "proxied"
class.

More importantly, ``FitsProviderProxy`` keeps an internal mapping of the sliced
extensions. So, we may be referring to ``sliced[0]`` and this would be mapped
to, say, ``nonsliced[3]``. ``FitsProviderProxy``.

Both ``FitsProvider`` and ``FitsProviderProxy`` can be studied as an example
implementation, but there is no need to follow them: please, evaluate carefully
the needs for your design, and feel free to depart from ours. As long as the
minimum interface is honored, AstroData will work as intended.

Note also that these classes were subject to heavy changes during development
and a future release cycle should see them refactored for clarity and to drop
any remnants of interfaces that were deprecated before the initial DRAGONS
public release.

As a last comment: remember that ``AstroData`` exposes its underlying
``DataProvider`` interface up to a certain point. This can be used to
dynamically expose to the user additional attributes, dependent on the
underlying technology, or even to the instrument, if needed. This is all fine
and encouraged **as long as everything is well documented**, and the user
understands that certain parts of the interface may not be available when using
different observatory's files\ [#soarnote]_, for example.

Registering a Data Provider to be Used with AstroData
=====================================================

Once we have a new data provider class, we need to let AstroData know how to
use it, and which class will make use of it. Normally a new data provider will
be associated to a new second level AstroData class (ie. a direct descendant to
``AstroData``, and a sibling of ``AstroDataFits``). This does not have to be
always the case, though: if an observatory organizes their FITS files in a way
that significantly departs from Gemini's, then creating a separate data
provider may be justified, if it makes it easier to deal with the data.

There are no instances of this as of yet, but we've made a conscious effort
during the design phase to make as easy as possible to plug in new providers.
Future release of this document will address this topic.


.. rubric:: Footnotes

.. [#viewnote] For efficiency reasons, and to keep just one version of the
   data. The method may decide to return a sliced copy instead, but this is
   a design decision.

.. [#soarnote] At the time of writing this manual, SOAR
   `has extended <https://github.com/soar-telescope/dragons-soar/tree/master>`_
   DRAGONS for their own use, but they are using the core FITS capabilities as
   defined by Gemini's implementation.
