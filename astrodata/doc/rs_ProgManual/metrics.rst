.. metrics:

*******************
QA Metrics Database
*******************


Event Manager, ADCC, and Fitsstore
==================================

QA metrics are reported to the fitsstore and stored locally
in the file .adcc/adcc_events.jsa using json.  The local storage
is not permanent and will be deleted with a superclean --safe call.

If access to fitsstore is available, the ``adcc`` will look at the 
local storage only for recent events, results from older reductions 
will come from the fitsstore.  If the local storage is empty, the 
``adcc`` will look to the fitsstore to retrieve whatever is available.

This system is currently used to report and retrieve QA metrics.
The nighttime GUI is using this event reporting as a QA metrics database.

Since a tool like the nighttime GUI might get its values either from
the local event database or from the fitsstore, through the adcc, the 
content and structure of both sources must be identical.  The interface 
must be respected.

Interface
=========

Record structure
----------------
The record is a nested dictionary.

The first level of the structure can contain the keys:

* `bg`
* `cc`
* `iq`
* `metadata`
* `msgtype`
* `timestamp`

Each are defined in the next sections.  `bg`, `cc`, and `iq`
are optional and applicable only when the `msgtype` is `'qametrics'`.

The record will therefore look like this::

   {'bg' : the bg record,
    'cc' : the cc record,
    'iq' : the iq record,
    'metadata : the metadata record,
    'msgtype' : the msgtype record,
    'timestamp' : the timestamp record
   }


bg
--
This record contains information about a background (BG) measurement.

Example
^^^^^^^

A record for 'bg' looks like this::

   'bg': {'band': '50',
          'brightness': 21.095,
          'brightness_error': 0.017,
          'comment': [], 
          'requested': 20
         }


Field Definitions
^^^^^^^^^^^^^^^^^

The fields for the `bg` record are:

+--------------------+--------------------------------------------------------------------+
| `band`             | The measured BG band as a number.  QUESTION: can it be eg. 50,80?? |
|                    | ?? Apparently fitsstore can return a 'None' string.  Why not None? |
|                    +--------------------------------------------------------------------+
|                    | Type: string                                                       |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'band': '50'                                                     |
+--------------------+--------------------------------------------------------------------+
| `brightness`       | Measured background magnitude.                                     |
|                    +--------------------------------------------------------------------+
|                    | Type: float                                                        |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'brightness': 21.095                                             |
+--------------------+--------------------------------------------------------------------+
| `brightness_error` | Error on the brightness measurement.                               |
|                    +--------------------------------------------------------------------+
|                    | Type: float                                                        |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'brightness_error': 0.017                                        |
+--------------------+--------------------------------------------------------------------+
| `comment`          | Any comments associated with the measurement.                      |
|                    +--------------------------------------------------------------------+
|                    | Type: list of strings                                              |
|                    +--------------------------------------------------------------------+
|                    | Examples::                                                         |
|                    |                                                                    |
|                    |   'comment': ['WARNING: BG requirement not met at the 95%          |
|                    |              confidence level']                                    |
|                    |   'comment': []                                                    |
+--------------------+--------------------------------------------------------------------+
| `requested`        | PI-requested BG constraint (Any == 100).                           |
|                    +--------------------------------------------------------------------+
|                    | Type: integer                                                      |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'requested': 20                                                  |
+--------------------+--------------------------------------------------------------------+


cc
--
This API contains information about a cloud cover (CC) measurement.

Example
^^^^^^^

A record for 'cc' looks like this::

   'cc': {'band': '50,70',
          'comment': ['WARNING: CC requirement not met at the 95% confidence level.'],
          'extinction': 0.09,
          'extinction_error': 0.047,
          'requested': 50,
          'zeropoint': {'e2v 10031--23-05,10031-01-03,10031-18-04': {'error': 0.047, 
                                                                    'value': 28.28}}
         }

Field Defintions
^^^^^^^^^^^^^^^^

The fields for the `cc` record are:

+--------------------+--------------------------------------------------------------------+
| ??? `band`         | The measured CC band as number or a comma-separated list of numbers|
|                    | for when the measurement is in agreement within the error with     |
|                    | multiple bands.                                                    |
|                    +--------------------------------------------------------------------+
|                    | Type: string                                                       |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'band': '50,70'                                                  |
+--------------------+--------------------------------------------------------------------+
| `comment`          | Any comments associated with the measurement.                      |
|                    +--------------------------------------------------------------------+
|                    | Type: list of strings                                              |
|                    +--------------------------------------------------------------------+
|                    | Examples::                                                         |
|                    |                                                                    |
|                    |   'comment': ['WARNING: CC requirement not met at the 95%          |
|                    |              confidence level']                                    |
+--------------------+--------------------------------------------------------------------+
| `extinction`       | Atmospheric extinction measured in magnitudes.                     |
|                    +--------------------------------------------------------------------+
|                    | Type: float                                                        |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'extinction': 0.09                                               |
+--------------------+--------------------------------------------------------------------+
| `extinction_error` | Error measured on the atmospheric extinction, in magnitudes.       |
|                    +--------------------------------------------------------------------+
|                    | Type: float                                                        |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'extinction_error': 0.047                                        |
+--------------------+--------------------------------------------------------------------+
| `requested`        | PI-requested CC constraint (Any == 100).                           |
|                    +--------------------------------------------------------------------+
|                    | Type: integer                                                      |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'requested': 50                                                  |
+--------------------+--------------------------------------------------------------------+
| `zeropoint`        | Zeropoint calculated from `extinction` and a nominal zeropoint,    |
|                    | with error estimate, for a specific `ampname` (or group of arrays) |
|                    +--------------------------------------------------------------------+
|                    | Type: dictionary                                                   |
|                    +--------------------------------------------------------------------+
|                    | Examples::                                                         |
|                    |                                                                    |
|                    |   'zeropoint': {'e2v 10031-23-05,10031-01-03,10031-18-04': \       |
|                    |                                                  {'error': 0.047,  |
|                    |                                                   'value': 28.28}  |
|                    |                }                                                   |
+--------------------+--------------------------------------------------------------------+




iq
--
This API contains information about a image quality (IQ) measurement.

Example
^^^^^^^

A record for 'iq' looks like this::

   'iq': { 'adaptive_optics': True,
           'ao_seeing': 1.224
           'ao_seeing_zenith': 1.176,
           'band': '85',
           'comment': ['WARNING: AO observation. IQ band from estimated AO seeing.'],
           'delivered': 0.854,
           'delivered_error': 0.022,
           'ellip_error': 0.036,
           'ellipticity': 0.017,
           'requested': 70,
           'strehl': 0.5,
           'zenith': 0.595,
           'zenith_error': 0.018
         }


Field Defintions
^^^^^^^^^^^^^^^^

The fields for the `iq` record are:

+--------------------+--------------------------------------------------------------------+
| `adaptive_optics`  | Identify the observation as an AO observation                      |
|  or is_ao ???      +--------------------------------------------------------------------+
|                    | Type: boolean                                                      |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'adaptive_optics': True                                          |
+--------------------+--------------------------------------------------------------------+
| `ao_seeing`        | Seeing reported by the AO system, as stored in the image header.   |
|                    | Optional, present only if `adaptive_optics` is `True`.             |
|                    +--------------------------------------------------------------------+
|                    | Type: float                                                        |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'ao_seeing': 1.224                                               |
+--------------------+--------------------------------------------------------------------+
| `ao_seeing_zenith` | `ao_seeing` corrected to zenith.  Optional, present only if        |
|      ???           | `adaptive_optics` is `True`.                                       |
|                    +--------------------------------------------------------------------+
|                    | Type: float                                                        |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'ao_seeing_zenith': 1.176                                        |
+--------------------+--------------------------------------------------------------------+
| `band`             | The measured IQ band as number.  QUESTION: can it be a list??      |
|                    +--------------------------------------------------------------------+
|                    | Type: string                                                       |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'band': '85'                                                     |
+--------------------+--------------------------------------------------------------------+
| `comment`          | Any comments associated with the measurement.                      |
|                    +--------------------------------------------------------------------+
|                    | Type: list of strings                                              |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'comment': ['WARNING: AO observation. IQ band from estimated     |
|                    |                AO seeing.']                                        |
+--------------------+--------------------------------------------------------------------+
| `delivered`        | Measured seeing in arcseconds.  Can be None for AO observations.   |
|                    +--------------------------------------------------------------------+
|                    | Type: float                                                        |
|                    +--------------------------------------------------------------------+
|                    | Examples::                                                         |
|                    |                                                                    |
|                    |   'delivered': None                                                |
|                    |   'delivered': 0.854                                               |
+--------------------+--------------------------------------------------------------------+
| `delivered_error`  | Error on the measurement of the delivered seeing.  Can be None     |
|                    | for AO observations.                                               |
|                    +--------------------------------------------------------------------+
|                    | Type: float                                                        |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'delivered_error': 0.022                                         |
+--------------------+--------------------------------------------------------------------+
| `ellip_error`      | Error on the ellipticity measurement.  Can be None for AO          |
|                    | observations.                                                      |
|                    +--------------------------------------------------------------------+
|                    | Type: float                                                        |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'ellip_error': 0.036                                             |
+--------------------+--------------------------------------------------------------------+
| `ellipticity`      | Measured ellipticity of sources used to measured seeing.  Can be   |
|                    | None for AO observations.                                          |
|                    +--------------------------------------------------------------------+
|                    | Type: float                                                        |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'ellipticity': 0.017                                             |
+--------------------+--------------------------------------------------------------------+
| `requested`        | PI-requested IQ constraint (Any == 100).                           |
|                    +--------------------------------------------------------------------+
|                    | Type: integer                                                      |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'requested': 70                                                  |
+--------------------+--------------------------------------------------------------------+
| `strehl`           | Measured Strehl ratio for AO observation.  Optional, present only  |
|   ???              | if `adaptive_optics` is `True`.                                    |
|                    +--------------------------------------------------------------------+
|                    | Type: float                                                        |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'strehl': 0.25                                                   |
+--------------------+--------------------------------------------------------------------+
| `zenith`           | `delivered` seeing corrected to zenith.  Can be None for AO        |
|                    | observations.                                                      |
|                    +--------------------------------------------------------------------+
|                    | Type: float                                                        |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'zenith': 0.595                                                  |
+--------------------+--------------------------------------------------------------------+
| `zenith_error`     | Error on the zenith-corrected seeing.  Can be None for AO          |
|                    | observations.                                                      |
|                    +--------------------------------------------------------------------+
|                    | Type: float                                                        |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'zenith_error': 0.018                                            |
+--------------------+--------------------------------------------------------------------+



metadata
--------
This API contains information about the observation and its file.

Example
^^^^^^^

A record for 'metadata' looks like this::

   'metadata': { 'airmass': 1.456,
                 'datalabel': 'GN-2015B-Q-26-27-001',
                 'filter': 'r',
                 'instrument': 'GMOS-N',
                 'local_time': '03:07:55.200000',
                 'object': 'SDSSJ 1110+64',
                 'raw_filename': 'N20160108S0160.fits',
                 'types': ['GEMINI_NORTH',
                           'GMOS_N',
                           'GMOS_IMAGE',
                           'GEMINI',
                           'SIDEREAL',
                           'GMOS_NODANDSHUFFLE',
                           'ACQUISITION',
                           'IMAGE'
                           'GMOS',
                           'GMOS_RAW',
                           'RAW',
                           'UNPREPARED'],
                 'ut_time': '2016-01-08 13:07:55.700000',
                 'waveband': 'r',
                 'wavelength': None
               }

Field Defintions
^^^^^^^^^^^^^^^^

The fields for the `metadata` record are:

+--------------------+--------------------------------------------------------------------+
| `airmass`          | Airmass of the observation.                                        |
|                    +--------------------------------------------------------------------+
|                    | Type: float                                                        |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'airmass': 1.456                                                 |
+--------------------+--------------------------------------------------------------------+
| `datalabel`        | Unique data identifier.                                            |
|                    +--------------------------------------------------------------------+
|                    | Type: string                                                       |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'datalabel': 'GN-2015B-Q-26-27-001'                              |
+--------------------+--------------------------------------------------------------------+
| `filter`           | Filter used for the observation.                                   |
|                    +--------------------------------------------------------------------+
|                    | Type: string                                                       |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'filter': 'r'                                                    |
+--------------------+--------------------------------------------------------------------+
| `instrument`       | Instrument used for the observation.                               |
|                    +--------------------------------------------------------------------+
|                    | Type: string                                                       |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'instrument': 'GMOS-N'                                           |
+--------------------+--------------------------------------------------------------------+
| `local_time`       | Local time of the observation. hh:mm:ss.ssssss                     |
|                    +--------------------------------------------------------------------+
|                    | Type: string                                                       |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'local_time': '03:07:55.200000'                                  |
+--------------------+--------------------------------------------------------------------+
| `object`           | Name of the target.                                                |
|                    +--------------------------------------------------------------------+
|                    | Type: string                                                       |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'object': 'SDSSJ 1110+64'                                        |
+--------------------+--------------------------------------------------------------------+
| `raw_filename`     | Name of the original, unprocessed observation file.                |
|                    +--------------------------------------------------------------------+
|                    | Type: string                                                       |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'raw_filename': 'N20160108S0160.fits'                            |
+--------------------+--------------------------------------------------------------------+
| `types`            | List of all applicable AstroDataTypes.                             |
|                    +--------------------------------------------------------------------+
|                    | Type: list of strings                                              |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'types': ['GEMINI_NORTH', 'GMOS_N', 'GMOS_IMAGE', 'GEMINI',      |
|                    |             'SIDEREAL', 'GMOS_NODANDSHUFFLE', 'ACQUISITION',       |
|                    |             'IMAGE', 'GMOS', 'GMOS_RAW', 'RAW', 'UNPREPARED']      |
+--------------------+--------------------------------------------------------------------+
| `ut_time`          | UT time of the observation.                                        |
|                    | Full datetime string format: YYYY-MM-DD hh:mm:ss.ssssss            |
|                    +--------------------------------------------------------------------+
|                    | Type: string                                                       |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'ut_time': '2016-01-08 13:07:55.700000'                          |
+--------------------+--------------------------------------------------------------------+
| `waveband`         | General wavelength band associated with the filter.                |
|                    +--------------------------------------------------------------------+
|                    | Type: string                                                       |
|                    +--------------------------------------------------------------------+
|                    | Example::                                                          |
|                    |                                                                    |
|                    |   'waveband': 'r'                                                  |
+--------------------+--------------------------------------------------------------------+
| `wavelength`       | ???                                                                |
|                    +--------------------------------------------------------------------+
|                    | Type: string                                                       |
|                    +--------------------------------------------------------------------+
|                    | Examples::                                                         |
|                    |                                                                    |
|                    |   'wavelength': None                                               |
|                    |   'wavelength': ???                                                |
+--------------------+--------------------------------------------------------------------+


msgtype
-------
This record defines the type of the event being reported. It is always stored as a `string`.
The valid events are:

`qametric`
   Applies to events reporting Quality Assessment metrics, like BG, CC, IQ.  Such
   events must contain at least one of the `bg`, `cc`, or `iq` records.

Any others???

Example::

   'msgtype': 'qametric'


timestamp
---------
This record contains the time of the event.  The value type is `float`.  
The format is ???

Example::

   'timestamp': 145225860.848475

Example of a full QA metrics record
===================================

Bringing it all together, here is an example of a complete QA metrics record.

::

   {'bg': { 'band': '50',
            'brightness': 21.123,
            'brightness_error': 0.018,
            'comment': [],
            'requested': 50},
    'cc': { 'band': '70',
            'comment': ['WARNING: CC requirement not met at the 95% confidence level'],
            'extinction': 0.086,
            'extinction_error': 0.063,
            'requested': 50,
            'zeropoint': {'e2v 10031-23-05,10031-01-03,10031-18-04': {'error': 0.063,
                                                                      'value': 28.284}}},
    'iq': { 'adaptive_optics': False,
            'band': '85',
            'comment': ['IQ requirement not met'],
            'delivered': 0.935,
            'delivered_error': 0.03,
            'ellip_error': 0.035,
            'ellipticity': 0.041,
            'requested': 70,
            'zenith': 0.873535978922236,
            'zenith_error': 0.028027892371836446},
    'metadata': { 'airmass': 1.12,
                  'datalabel': 'GN-2015B-FT-26-16-002',
                  'filter': 'r',
                  'instrument': 'GMOS-N',
                  'local_time': '19:21:58.200000',
                  'object': 'L91',
                  'raw_filename': 'N20160108S0056.fits',
                  'types': [ 'GEMINI_NORTH',
                             'GMOS_N',
                             'GMOS_IMAGE',
                             'GEMINI',
                             'SIDEREAL',
                             'IMAGE',
                             'GMOS',
                             'GMOS_RAW',
                             'RAW',
                             'UNPREPARED'],
                  'ut_time': '2016-01-08 05:21:58.700000',
                  'waveband': 'r',
                  'wavelength': None},
    'msgtype': 'qametric',
    'timestamp': 1452231232.572889}
   }
