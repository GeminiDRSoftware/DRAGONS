AstroDataIGRINS2
================

AstroData class for the IGRINS-2 instrument.

This class provides instrument-specific tags and descriptors for the
IGRINS-2 instrument, handling data from the Immersion Grating Infrared
Spectrograph (IGRINS-2) instrument.

Tags
----
The following tags are defined for IGRINS-2 data:

From AstroDataIGRINS2:
- 'IGRINS', 'IGRINS-2': Basic instrument identification
  Derived from: INSTRUME header keyword
- 'ARC', 'CAL': For arc lamp calibration frames (sky observations)
  Derived from: OBSTYPE='OBJECT' and 'sky' in OBJECT header
- 'FLAT', 'CAL': For flat field calibration frames
  Derived from: OBSTYPE='FLAT'
- 'LAMPON'/'LAMPOFF': For calibration lamp status
  Derived from: GCALLAMP and GCALSHUT headers (blocked if processed)
- 'SKY', 'CAL': For sky observations
  Derived from: OBSTYPE='OBJECT' and 'sky' in OBJECT (blocked if processed as arc)
- 'STANDARD', 'CAL': For standard star observations
  Derived from: OBSCLASS='partnerCal' and 'sky' not in OBJECT

From AstroDataIGRINSBase:
- 'IGRINS', 'IGRINS-1': Base instrument identification
  Derived from: INSTRUME header keyword
- 'FORCED': For manually forced tags
  Derived from: TAG_FORCED header keyword

Descriptors
-----------
The following descriptors are available:

From AstroDataIGRINS2:
- instrument(generic=False): Returns the instrument name
  Returns: 'IGRINS-2' or 'IGRINS' if generic=True
  Derived from: INSTRUME header
- band(): Returns the filter/wavelength band (H or K)
  Returns: str (e.g., 'H' or 'K')
  Derived from: FILTER header
- ut_datetime(): Returns the observation datetime
  Returns: datetime object
  Derived from: UTDATETIME or UTSTART header

From AstroDataIGRINSBase:
- read_noise(): Returns the read noise in electrons
  Returns: float/list of read noise values
  Derived from: NSAMP header and band-specific lookup table
- arm(): Returns the spectrograph arm (H or K band)
  Returns: str ('H' or 'K')
  Derived from: Wavelength band header (BAND or FILTER)
- array_section(pretty=False): Returns the rectangular section of exposed pixels
  Returns: Section object or string
  Default: [1:2048,1:2048] for full frame
- data_section(pretty=False): Returns the data section of the detector
  Returns: Section object or string
  Default: [1:2048,1:2048] for full frame
- detector_section(pretty=False): Returns the detector section
  Returns: Section object or string
  Default: [1:2048,1:2048] for full frame
- wcs_ra(): Returns RA from WCS
  Returns: float (right ascension in degrees)
  Derived from: CRVAL1 header
- wcs_dec(): Returns Dec from WCS
  Returns: float (declination in degrees)
  Derived from: CRVAL2 header
- exposure_time(): Returns the exposure time
  Returns: float
  Derived from: EXPTIME header
- observation_class(): Returns the observation class
  Returns: str (e.g., 'science', 'acq', 'projCal')
  Derived from: BAND and other headers
- observation_type(): Returns the observation type
  Returns: str (e.g., 'OBJECT', 'DARK', 'FLAT_OFF')
  Derived from: OBSTYPE header

Common Header Keywords Used
--------------------------
- INSTRUME: Identifies the instrument ('IGRINS-2')
- OBSTYPE: Determines observation type
- OBJECT: Object name (identifies sky observations)
- OBSCLASS: Observation class
- GCALLAMP/GCALSHUT: Calibration lamp status
- FILTER/BAND: Wavelength band (H or K)
- EXPTIME: Exposure time in seconds
- UTDATETIME/UTSTART: Observation timestamp
- CRVAL1/2: WCS reference coordinates
- NSAMP: Number of Fowler samples for read noise calculation





.. autoclass:: igrins_instruments.igrins.adclass.AstroDataIGRINS2
   :members:
