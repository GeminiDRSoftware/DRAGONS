from astrodata import astro_data_tag, astro_data_descriptor, returns_list, TagSet
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

    # @returns_list
    # @astro_data_descriptor
    # def gain(self):
    #     """
    #     Returns the gain (electrons/ADU) from lookup table
    #
    #     Returns
    #     -------
    #     float/list
    #         gain
    #     """
    #     return lookup.array_properties.get('gain')
    #
    # @returns_list
    # @astro_data_descriptor
    # def read_noise(self):
    #     """
    #     Returns the read_noise electron from lookup table
    #
    #     Returns
    #     -------
    #     float/list
    #         gain
    #     """
    #     return lookup.array_properties.get('read_noise')


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

