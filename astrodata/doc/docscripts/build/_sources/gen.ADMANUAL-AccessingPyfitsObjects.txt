


Access to Pyfits and Numpy Objects
----------------------------------

Access to pyfits objects used by AstroData internally is, technically
allowed. In general, it is possible to do so without any serious harm,
but changes to these structures can possibly affect operation of the
AstroData instance which contains it. To obtain the HDUList associated
with an AstroData instance one merely accesses the "hdulist" member,
which contains a pyfits HDUList object. Note, every AstroData instance
has it's own unique HDUList object. Sub-data shares HDUs with the data
it was sliced from (i.e. ad["SCI"] contains HDUs which also appear in
ad, assuming there is at least one extension with EXTNAME="SCI", but
ad["SCI"].hdulist will not be the same object as ad.hdulist).

Reasons to access the hdulist are limited and in general one does not
need the HDUList directly since the AstroData encompasses the list-
like behavior of the MEF. Similarly, one doesn't generally require
access to an HDU, since single-HDU AstroData instances behave like the
HDU, having "data" and "header" members which constitute the two
aspects of an HDU.


Pyfits Header
-------------

Note, for a single-HDU AstroData instance, "ad", "ad.header" is the
pyfits.Header object. One may want this, for example, to pass to the
contructor of a new AstroData instance. One might also want to set
key-value pairs in the header directly. But this case it is better to
use the AstroData member functions, AstroData.setKeyValue(..) (or more
generally the AstroData.xxxSetKeyValue and AstroData.xxxGetKeyValue,
where "xxx" is either nothing, for single-HDU AstroData instances,
"phu" for PHU settings, and "ext" for setting where the extension
header intended is specified in an additional argument. The reason is
that changes to the header can affect type information, and use of
AstroData allows the system to try to keep information up to data,
such as types which are dependent on header settings.

Note: currently the one required use for the pyfits Header structure
is if one seeks to create or append to an AstroData instance by giving
a header and data objects. It's possible we should remove this one
example by supporting use of dictionaries for this purpose. The reason
this was not done yet is due to the comments... a header is not merely
a key-value structure, aka, a dictionary, but also has a second value,
the comment.


Numpy Ndarray
-------------

AstroData does not attempt to proxy or represent numpy structures. In
general the ad.data member for a single-HDU AstroData instance will be
a numpy ndarray. The user is meant to manipulate this entirely on
their own, and to keep track of how this array might be shared by
various AstroData instances.

