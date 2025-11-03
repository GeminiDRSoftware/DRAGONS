import datetime
"""
This module defines AstroData classes for IGRINS instruments, extending
`gemini_instruments.igrins.AstroDataIgrins` to provide instrument-specific
metadata handling and data descriptors for IGRINS and IGRINS-2 data.
"""

from astropy.nddata import NDData
import astropy.units as u
import numpy as np
from astrodata import (astro_data_tag, astro_data_descriptor,
                       returns_list, TagSet)
from gemini_instruments import gmu
from gemini_instruments.common import Section
from . import lookup
from gemini_instruments import igrins

# Since gemini_instruments already defines AstroDataIgrins which is picked up
# by the mapper, we create a new AstroDataIGRINS inheriting from the
# gemini_instruments.

class AstroDataIGRINS_(igrins.AstroDataIgrins):
    """
    Base AstroData class for IGRINS instruments.

    This class extends `gemini_instruments.igrins.AstroDataIgrins` and provides
    common keyword mappings and descriptor overrides for IGRINS data.
    """
    # This is a temporary placeholder class. The original gemini_instruments
    # version of AstroDataIgrins could be updated with it contents eventually.

    __keyword_dict = dict(
        exposure_time = "EXPTIME",
        airmass = 'AMSTART',
        wavelength_band = 'BAND',
        detector_name = 'DETECTOR',
        target_name = 'OBJECT',
        observation_class = 'OBSCLASS',
        observation_type = 'OBJTYPE',
        ra = 'OBJRA',
        dec = 'OBJDEC',
        slit_x_center = 'SLIT_CX',
        slit_y_center = 'SLIT_CY',
        slit_width = 'SLIT_WID',
        slit_length = 'SLIT_LEN',
        slit_angle = 'SLIT_ANG'
    )

    @astro_data_tag
    def _tag_bundle(self):
        # Gets blocked by tags created by split files
        return TagSet(['BUNDLE'])

    @astro_data_tag
    def _tag_spect(self):
        bands = set(self.hdr.get('BAND'))
        if len(bands) == 1:
            return TagSet([bands.pop(), 'SPECT'], blocks=['BUNDLE'])

    # LAMPON/lAMPOFF tags are inccorectly set by AstroDataGemini._type_gacl_alp
    # method. We simply override this to return nothing.
    @astro_data_tag
    def _type_gcal_lamp(self):
        return


    @astro_data_descriptor
    def observation_class(self):
        """
        Returns 'class' the observation; one of,

            'science', 'acq', 'projCal', 'dayCal', 'partnerCal', 'acqCal'

        An 'acq' is defined by BAND == 'S', where 'S' indicates a slit image.

        Returns
        -------
        oclass: <str>
            One of the above enumerated names for observation class.

        """
        oclass = None

        otype = self.phu.get(self._keyword_for('observation_class'))
        if not otype:
            otype = self[0].hdr.get(self._keyword_for('observation_class'))

        if 'S' in self.wavelength_band():
            oclass = 'acq'

        if isinstance(otype, str):
            if 'STD' in otype:
                oclass = 'partnerCal'
            elif 'TAR' in otype:
                oclass = 'science'
            elif otype in ["partnerCal"]:
                oclass = 'partnerCal'
        else:
            oclass = "unknown"

        return oclass

    @astro_data_descriptor
    def observation_type(self):
        """
        Returns 'type' the observation. For IGRINS, this will be one of,

            'OBJECT', 'DARK', 'FLAT_OFF', 'FLAT_ON', 'ARC'

        Returns
        -------
        otype: <str>
            Observation type.

        """
        otype = self.phu.get(self._keyword_for('observation_type'))
        if not otype:
            otype = self[0].hdr.get(self._keyword_for('observation_type'))
        ftype = self.phu.get("FRMTYPE")
        if not otype:
            ftype = self[0].hdr.get("FRMTYPE")

        if otype in ['STD', 'TAR']:
            otype = 'OBJECT'
        elif otype in ['FLAT']:
            otype = f"FLAT_{ftype}"

        return otype

# This will be shared between IGRINS and IGRINS2

class AstroDataIGRINSBase(AstroDataIGRINS_):
    """
    AstroData base class for IGRINS and IGRINS-2 instruments.

    This class provides common descriptors and tags applicable to both IGRINS
    and IGRINS-2 data, such as `read_noise`, `array_section`, and `wcs_ra`/`wcs_dec`.
    """
    # single keyword mapping.  add only the ones that are different
    # from what's already defined in AstroDataGemini.

    @astro_data_tag
    def _tag_forcced(self):
        # workaround for wrong headers. Check fixHeader recipe.
        tag_forced = self.phu.get('TAG_FORCED', '')
        return TagSet(tag_forced.split())

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['IGRINS', 'IGRINS-1'])

    # ------------------
    # Common descriptors
    # ------------------

    # def wcs(self):
    #     return object()

    @returns_list
    @astro_data_descriptor
    def read_noise(self):
        """
        Returns the read noise in electrons.

        Returns
        -------
        float/list
            readnoise
        """
        if self.is_single:
            fowler_samp =self.hdr.get('NSAMP') 
            read_noise_fit = lookup.array_properties.get("read_noise_fit")[self.band()]
            read_noise = np.polyval(read_noise_fit, 1/fowler_samp)
        else:
            read_noise = [ext.read_noise() for ext in self]

        return read_noise

    # FIXME We are hardcoding array_section, detector_section, data_section.
    # Not sure if this is wise thing to do.

    @astro_data_descriptor
    def arm(self):
        return self.phu.get(self._keyword_for('wavelength_band'))

    # copied from f2
    @returns_list
    @astro_data_descriptor
    def array_section(self, pretty=False):
        """
        Returns the rectangular section that includes the pixels that would be
        exposed to light.  If pretty is False, a tuple of 0-based coordinates
        is returned with format (x1, x2, y1, y2).  If pretty is True, a keyword
        value is returned without parsing as a string.  In this format, the
        coordinates are generally 1-based.

        One tuple or string is return per extension/array, in a list. If the
        method is called on a single slice, the section is returned as a tuple
        or a string.

        Parameters
        ----------
        pretty : bool
         If True, return the formatted string found in the header.

        Returns
        -------
        tuple of integers or list of tuples
            Location of the pixels exposed to light using Python slice values.

        string or list of strings
            Location of the pixels exposed to light using an IRAF section
            format (1-based).
        """
        # Since none of the parent class defines array_section, we simply skip.
        # if 'PREPARED' in self.tags:
        #     return super().array_section(pretty=pretty)

        value_filter = (str if pretty else Section.from_string)
        return value_filter('[1:2048,1:2048]')

    # copied from f2
    @returns_list
    @astro_data_descriptor
    def data_section(self, pretty=False):

        value_filter = (str if pretty else Section.from_string)
        return value_filter('[1:2048,1:2048]')

    # copied from f2
    @returns_list
    @astro_data_descriptor
    def detector_section(self, pretty=False):

        value_filter = (str if pretty else Section.from_string)
        return value_filter('[1:2048,1:2048]')


    @astro_data_descriptor
    def wcs_ra(self):
        """
        Returns the Right Ascension of the center of the field based on the
        WCS rather than the RA keyword. This just uses the CRVAL1 keyword.

        Returns
        -------
        float
            right ascension in degrees
        """
        # Try the first (only if sliced) extension, then the PHU
        try:
            h = self[0].hdr
            crval = h['CRVAL1']
            ctype = h['CTYPE1']
        except (KeyError, IndexError):
            crval = self.phu.get('CRVAL1')
            ctype = self.phu.get('CTYPE1')
        # return crval if ctype == 'RA---TAN' else None
        return 1

    @astro_data_descriptor
    def wcs_dec(self):
        """
        Returns the Declination of the center of the field based on the
        WCS rather than the DEC keyword. This just uses the CRVAL2 keyword.

        Returns
        -------
        float
            declination in degrees
        """
        # Try the first (only if sliced) extension, then the PHU
        try:
            h = self[0].hdr
            crval = h['CRVAL2']
            ctype = h['CTYPE2']
        except (KeyError, IndexError):
            crval = self.phu.get('CRVAL2')
            ctype = self.phu.get('CTYPE2')
        #return crval if ctype == 'DEC--TAN' else None
        return 1

    @astro_data_descriptor
    def exposure_time(self):
        return self.hdr["EXPTIME"]


class AstroDataIGRINS(AstroDataIGRINSBase):
    """
    AstroData class for the original IGRINS instrument.

    This class provides instrument-specific tags and descriptors for the
    original IGRINS instrument, including `data_label`, `program_id`, and `band`.
    """
    # single keyword mapping.  add only the ones that are different
    # from what's already defined in AstroDataGemini.

    #@astro_data_tag
    def _tag_caltype(self):
        if self.phu.get('OBJTYPE') == 'SKY' or self[0].hdr.get('OBJTYPE') == 'SKY':
            return TagSet(['SKY', 'ARC'])
        elif self.phu.get('FRMTYPE') == "ON":
            return TagSet(['LAMPON'])
        elif self.phu.get('FRMTYPE') == "OFF":
            return TagSet(['LAMPOFF'])

    @astro_data_descriptor
    def data_label(self):

        # The IGRINS data does not have a DATALAB keyword inthe header. Thus,
        # the `data_label` descriptor return None, which raises an error during
        # storeProcessedDark. We try to define a ad-hoc data label out of its
        # file name. But it should be revisited. FIXME

        # The header has 'GEMPRID' etc, but no OBSID and such is defined.
        _, obsdate, obsid = self.filename.split('.')[0].split('_')[:3]
        datalab = f"igrins-{obsdate}-{obsid}"
        return datalab

    @astro_data_descriptor
    def program_id(self):
        """
        Returns the ID of the program the observation was taken for

        Returns
        -------
        str
            the program ID
        """
        progid = self.phu.get('GEMPRGID')
        if progid is None:
            progid = "GN-2099A-Q-000"

        return progid

    @astro_data_descriptor
    def band(self):
        return self.phu.get('BAND')

class AstroDataIGRINS2(AstroDataIGRINSBase):
    """
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
    """

    __keyword_dict = dict(
        wavelength_band = 'FILTER',
    )

    @staticmethod
    def _matches_data(source):
        igrins = source[0].header.get('INSTRUME', '').upper() == 'IGRINS-2'
        if not igrins:
            igrins = source[1].header.get('INSTRUME', '').upper() == 'IGRINS-2'

        return igrins

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['IGRINS', 'IGRINS-2'])

    @astro_data_tag
    def _tag_arc(self):
        if (self.phu.get("OBSTYPE") == "OBJECT" and
                "sky" in self.phu.get("OBJECT", '').lower()):
            return TagSet(['ARC', 'CAL'], if_present=['PROCESSED'])

    @astro_data_tag
    def _tag_flat(self):
        if self.phu.get('OBSTYPE').strip() == 'FLAT':
            return TagSet(['FLAT', 'CAL'])

    @astro_data_tag
    def _type_gcal_lamp(self):
        # When flats are processed, they're neither "on" nor "off"
        if self.phu.get('GCALLAMP') == 'QH' and self.phu.get('GCALSHUT') == 'CLOSED':
            return TagSet(['LAMPON'], blocked_by=['PROCESSED'])
        elif self.phu.get('GCALLAMP') == 'IRhigh' and self.phu.get('GCALSHUT') == 'CLOSED':
            return TagSet(['LAMPOFF'], blocked_by=['PROCESSED'])

    @astro_data_tag
    def _tag_sky(self):
        if (self.phu.get("OBSTYPE") == "OBJECT" and
                "sky" in self.phu.get("OBJECT").lower()):
            # We don't want "SKY" if it's become a processed arc
            return TagSet(['SKY', 'CAL'], blocked_by=['PROCESSED'])

    @astro_data_tag
    def _tag_std(self):
        if (self.phu.get("OBSTYPE") == "OBJECT" and
                self.phu.get("OBSCLASS") == "partnerCal" and
                not "sky" in self.phu.get('OBJECT', '').lower()):
            return TagSet(['STANDARD', 'CAL'], blocked_by=['SKY', 'CAL'])

    #@astro_data_tag -- commented out so not run
    def _tag_caltype(self):
        tags = TagSet()

        if self.phu.get("OBSTYPE").strip() == "FLAT":
            tags.add.add("FLAT")
        elif self.phu.get("OBSTYPE") == "OBJECT" and self.phu.get("OBSCLASS") == "partnerCal":
            if "sky" in self.phu.get("OBJECT").lower():
                tags.add.add("SKY")
                tags.add.add("ARC")
            else:
                tags.add.add("STANDARD")
        else:
            tags.add.add("SCIENCE")

        if self.phu.get("GCALLAMP") == "QH" and self.phu.get("GCALSHUT") == "CLOSED":
            tags.add.add("LAMPON")
        elif self.phu.get("GCALLAMP") == "IRhigh" and self.phu.get("GCALSHUT") == "CLOSED":
            tags.add.add("LAMPOFF")

        return tags

    @astro_data_tag
    def _tag_spect(self):
        band = self.phu.get('FILTER')
        if band:
            return TagSet([band, 'SPECT'], blocks=['BUNDLE'])

    @astro_data_descriptor
    def instrument(self, generic=False):
        """
        Returns the name of the instrument making the observation

        Parameters
        ----------
        generic: boolean
            If set, don't specify the specific instrument if there are clones
            (e.g., return "IGRINS" rather than "IGRINS-2")

        Returns
        -------
        str
            instrument name
        """
        return 'IGRINS' if generic else self.phu.get('INSTRUME')

    @astro_data_descriptor
    def band(self):
        return self.phu.get('FILTER')

    @staticmethod
    def _get_udatetime(hdr, dateonly=False, timeonly=False):
        utdatetime = hdr.get('UTDATETI', None)
        if utdatetime is None:
            utdatetime = hdr.get('UTSTART', None)

        if utdatetime is None:
            raise KeyError("The header needs UTDATETIME or UTSART")

        dt = datetime.datetime.fromisoformat(utdatetime)

        if dateonly:
            return dt.date()
        elif timeonly:
            return dt.time()
        else:
            return dt

    @astro_data_descriptor
    def ut_datetime(self, strict=False, dateonly=False, timeonly=False):
        # FIXME To workaround an issue in dragons4, which try to do
        # ad.phu['UTSTART'] (primitive_gemini.py:244), we have a primitive that
        # temporarily rename UTSTART to UTDATETIME. This is a work around for thos cases.

        if self.is_single:
            return self._get_udatetime(self.hdr, dateonly=dateonly, timeonly=timeonly)
        else:
            try:
                return self._get_udatetime(self.phu, dateonly=dateonly, timeonly=timeonly)
            except KeyError:
                if len(self):
                    return self._get_udatetime(self[0].hdr,
                                               dateonly=dateonly, timeonly=timeonly)


