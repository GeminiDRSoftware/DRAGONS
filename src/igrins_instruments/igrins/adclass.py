from astrodata import astro_data_tag, astro_data_descriptor, returns_list, TagSet
from gemini_instruments import gmu
from gemini_instruments.common import Section
from . import lookup
from gemini_instruments import igrins

    # igrins.AstroDataIgrins as _AstroDataIgrins)

class _AstroDataIGRINS(igrins.AstroDataIgrins):
    """fix to bug in gemini_instrument module
    """
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

            'OBJECT', 'DARK', 'FLAT', 'ARC'

        Returns
        -------
        otype: <str>
            Observation type.

        """
        otype = self.phu.get(self._keyword_for('observation_type'))
        if not otype:
            otype = self[0].hdr.get(self._keyword_for('observation_type'))

        if otype in ['STD', 'TAR']:
            otype = 'OBJECT'
        elif otype in ['FLAT']:
            otype = 'FLAT'
        elif otype in ['DARK']:
            otype = 'DARK'
        elif otype in ['ARC']:
            otype = 'ARC'
        elif otype in ['SKY']:
            otype = 'SKY'

        return otype

class AstroDataIGRINS(_AstroDataIGRINS):
    # single keyword mapping.  add only the ones that are different
    # from what's already defined in AstroDataGemini.

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['IGRINS', 'VERSION1'])


    @astro_data_tag
    def _tag_flat(self):
        #if self.phu.get('SOMEKEYWORD') == 'Flat_or_something':
        #    return TagSet(['FLAT', 'CAL']])
        pass

    # ------------------
    # Common descriptors
    # ------------------

    @astro_data_descriptor
    def gain(self):
        """
        Returns the gain (electrons/ADU) from lookup table

        Returns
        -------
        float/list
            gain
        """
        return lookup.array_properties.get('gain')


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

