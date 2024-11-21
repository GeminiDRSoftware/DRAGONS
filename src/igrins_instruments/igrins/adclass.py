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

class _AstroDataIGRINS(igrins.AstroDataIgrins):
    # This is a temporary placeholder class. The original gemini_instruments
    # version of AstroDataIgrins could be updated with it contents eventually.

    __keyword_dict = dict(
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

class AstroDataIGRINSBase(_AstroDataIGRINS):
    # single keyword mapping.  add only the ones that are different
    # from what's already defined in AstroDataGemini.

    @astro_data_tag
    def _tag_forcced(self):
        # workaround for wrong headers. Check fixHeader recipe.
        tag_forced = self.phu.get('TAG_FORCED', '')
        return TagSet(tag_forced.split())

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['IGRINS', 'VERSION1'])

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

class AstroDataIGRINS(AstroDataIGRINSBase):
    # single keyword mapping.  add only the ones that are different
    # from what's already defined in AstroDataGemini.

    @astro_data_tag
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
        return TagSet(['IGRINS', 'VERSION2'])

    @astro_data_tag
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
        bands = set(self.hdr.get('FILTER'))
        if len(bands) == 1:
            return TagSet([bands.pop(), 'SPECT'], blocks=['BUNDLE'])

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
        return self.hdr.get('FILTER')
