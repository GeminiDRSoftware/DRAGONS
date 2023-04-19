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
    def _tag_sky(self):
        if self.phu.get('OBJTYPE') == 'SKY' or self[0].hdr.get('OBJTYPE') == 'SKY':
            return TagSet(['SKY', 'CAL'])

    @astro_data_tag
    def _tag_lamp(self):
        if self.phu.get('FRMTYPE') == "ON":
            return TagSet(['LAMPON'])
        elif self.phu.get('FRMTYPE') == "OFF":
            return TagSet(['LAMPOFF'])

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

        if 'STD' in otype:
            oclass = 'partnerCal'
        elif 'TAR' in otype:
            oclass = 'science'
        elif otype in ["partnerCal"]:
            oclass = 'partnerCal'

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

class AstroDataIGRINS(_AstroDataIGRINS):
    # single keyword mapping.  add only the ones that are different
    # from what's already defined in AstroDataGemini.

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['IGRINS', 'VERSION1'])

    # ------------------
    # Common descriptors
    # ------------------

    def wcs(self):
        return object()

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
        return lookup.array_properties.get('read_noise')

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



class AstroDataIGRINS2(AstroDataIGRINS):
    # single keyword mapping.  add only the ones that are different
    # from what's already defined in AstroDataGemini.

    @staticmethod
    def _matches_data(source):
        grins = source[0].header.get('INSTRUME', '').upper() == 'IGRINS-2'
        if not grins:
            grins = source[1].header.get('INSTRUME', '').upper() == 'IGRINS-2'

        return grins

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['IGRINS', 'VERSION2'])

