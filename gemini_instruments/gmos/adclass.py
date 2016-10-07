import re

from astrodata import astro_data_tag, astro_data_descriptor, returns_list, TagSet

from ..gemini import AstroDataGemini
from .. import gmu
from . import lookup

class AstroDataGmos(AstroDataGemini):
    @staticmethod
    def _matches_data(data_provider):
        return data_provider.phu.get('INSTRUME', '').upper() in ('GMOS-N', 'GMOS-S')

    @astro_data_tag
    def _tag_instrument(self):
        # tags = ['GMOS', self.instrument().upper().replace('-', '_')]
        return TagSet(['GMOS'])

    @astro_data_tag
    def _tag_dark(self):
        if self.phu.get('OBSTYPE') == 'DARK':
            return TagSet(['DARK'], blocks=['IMAGE', 'SPECT'])

    @astro_data_tag
    def _tag_arc(self):
        if self.phu.get('OBSTYPE') == 'ARC':
            return TagSet(['ARC', 'CAL'])

    @astro_data_tag
    def _tag_bias(self):
        if self.phu.get('OBSTYPE') == 'BIAS':
            return TagSet(['BIAS', 'CAL'], blocks=['IMAGE', 'SPECT'])

    @astro_data_tag
    def _tag_flat(self):
        if self.phu.get('OBSTYPE') == 'FLAT':
            if self.phu.get('GRATING') == 'MIRROR':
                f1, f2 = self.phu.FILTER1, self.phu.FILTER2
                # This kind of filter prevents imaging to be classified as FLAT
                if any(('Hartmann' in f) for f in (f1, f2)):
                    return

            return TagSet(['GCALFLAT', 'FLAT', 'CAL'])

    @astro_data_tag
    def _tag_twilight(self):
        if self.phu.get('OBJECT').upper() == 'TWILIGHT':
            # Twilight flats are of OBSTYPE == OBJECT, meaning that the generic
            # FLAT tag won't be triggered. Add it explicitly
            return TagSet(['TWILIGHT', 'CAL', 'FLAT'])

    def _tag_is_spect(self):
        pairs = (
            ('MASKTYP', 0),
            ('MASKNAME', 'None'),
            ('GRATING', 'MIRROR')
        )

        matches = (self.phu.get(kw) == value for (kw, value) in pairs)
        if any(matches):
            return False
        return True

    @astro_data_tag
    def _tag_ifu(self):
        if not self._tag_is_spect():
            return

        mapping = {
            'IFU-B': 'ONESLIT_BLUE',
            'IFU-B-NS': 'ONESLIT_BLUE',
            'b': 'ONESLIT_BLUE',
            'IFU-R': 'ONESLIT_RED',
            'IFU-R-NS': 'ONESLIT_RED',
            'r': 'ONESLIT_RED',
            'IFU-2': 'TWOSLIT',
            'IFU-NS-2': 'TWOSLIT',
            's': 'TWOSLIT'
        }

        names = set(key for key in mapping.keys() if key.startswith('IFU'))

        mskt, mskn = self.phu.get('MASKTYP'), self.phu.get('MASKNAME')
        if mskt == -1 and (mskn in names or re.match('g[ns]ifu_slit[rbs]_mdf', mskn)):
            if mskn not in names:
                mskn = re.match('g.ifu_slit(.)_mdf', mskn).groups()[0]

            return TagSet(['SPECT', 'IFU', mapping[mskn]])

    @astro_data_tag
    def _tag_mask(self):
        spg = self.phu.get
        if spg('GRATING') == 'MIRROR' and spg('MASKTYP') != 0:
            return TagSet(['MASK'])

    @astro_data_tag
    def _tag_image(self):
        if self.phu.get('GRATING') == 'MIRROR':
            return TagSet(['IMAGE'])

    @astro_data_tag
    def _tag_ls(self):
        if not self._tag_is_spect():
            return

        if self.phu.get('MASKTYP') == 1 and self.phu.get('MASKNAME', '').endswith('arcsec'):
            return TagSet(['SPECT', 'LS'])

    @astro_data_tag
    def _tag_mos(self):
        if not self._tag_is_spect():
            return

        mskt = self.phu.get('MASKTYP')
        mskn = self.phu.get('MASKNAME', '')
        if mskt == 1 and not (mskn.startswith('IFU') or mskn.startswith('focus') or mskn.endswith('arcsec')):
            return TagSet(['SPECT', 'MOS'])

    @astro_data_tag
    def _tag_nodandshuffle(self):
        if 'NODPIX' in self.phu:
            return TagSet(['NODANDSHUFFLE'])

    @property
    def instrument_name(self):
        return 'GMOS'

    @astro_data_descriptor
    def amp_read_area(self):
        pass

    @astro_data_descriptor
    def array_name(self):
        pass

    @astro_data_descriptor
    def central_wavelength(self, asMicrometers=False, asNanometers=False, asAngstroms=False):
        """
        Returns the central wavelength in meters or specified units

        Returns
        -------
        float
            The central wavelength setting
        """
        pass

    @astro_data_descriptor
    def detector_name(self):
        pass

    @astro_data_descriptor
    def detector_rois_requested(self):
        pass

    @astro_data_descriptor
    def detector_x_bin(self):
        """
        Returns the detector binning in the x-direction

        Returns
        -------
        int
            The detector binning
        """
        pass

    @astro_data_descriptor
    def detector_y_bin(self):
        """
        Returns the detector binning in the y-direction

        Returns
        -------
        int
            The detector binning
        """
        pass

    @astro_data_descriptor
    def disperser(self, stripID=False, pretty=False):
        pass

    @astro_data_descriptor
    def dispersion(self, asMicrometers=False, asNanometers=False, asAngstroms=False):
        pass

    @astro_data_descriptor
    def dispersion_axis(self):
        pass

    @astro_data_descriptor
    def exposure_time(self):
        """
        Returns the exposure time in seconds.

        Returns
        -------
        float
            Exposure time.

        """
        pass

    @astro_data_descriptor
    def focal_plane_mask(self, stripID=False, pretty=False):
        """
        Returns the name of the focal plane mask.  The component ID can be
        removed with either 'stripID' or 'pretty' set to True.

        Parameters
        ----------
        stripID : bool
            If True, removes the component ID and returns only the name of
            the focal plane mask.
        pretty : bool
            Same as for stripID.  Pretty here does not do anything more.

        Returns
        -------
        str
            The name of the focal plane mask with or without the component ID.

        """
        pass

    @returns_list
    @astro_data_descriptor
    def gain(self):
        """
        Returns the gain (electrons/ADU) for each extension

        Returns
        -------
        list
            Gains used for the observation

        """
        pass

    @astro_data_descriptor
    def gain_setting(self):
        pass

    @astro_data_descriptor
    def group_id(self):
        """
        Returns a string representing a group of data that are compatible
        with each other.  This is used when stacking, for example.  Each
        instrument and mode of observation will have its own rules.

        At the Gemini class level, the default is to group by the Gemini
        observation ID.

        Returns
        -------
        str
            A group ID for compatible data.

        """
        pass

    @astro_data_descriptor
    def nod_count(self):
        pass

    @astro_data_descriptor
    def nod_offset(self):
        pass

    @astro_data_descriptor
    def nod_pixels(self):
        pass

    @returns_list
    @astro_data_descriptor
    def nominal_photometric_zeropoint(self):
        """
        Returns the nominal zeropoints (i.e., the magnitude corresponding to
        a pixel value of 1) for the extensions in an AD object.

        Returns
        -------
        float/list
            zeropoint values, one per SCI extension
        """
        pass

    @returns_list
    @astro_data_descriptor
    def non_linear_level(self):
        """
        Returns the level at which the data become non-linear, in ADU.

        Returns
        -------
        float/list
            Value(s) at which the data become non-linear
        """
        pass

    @astro_data_descriptor
    def overscan_section(self, pretty=False):
        """
        Returns the overscan (or bias) section.  If pretty is False, a
        tuple of 0-based coordinates is returned with format (x1, x2, y1, y2).
        If pretty is True, a keyword value is returned without parsing as a
        string.  In this format, the coordinates are generally 1-based.

        One tuple or string is return per extension/array.  If more than one
        array, the tuples/strings are return in a list.  Otherwise, the
        section is returned as a tuple or a string.

        Parameters
        ----------
        pretty : bool
         If True, return the formatted string found in the header.

        Returns
        -------
        tuple of integers or list of tuples
            Position of the overscan section using Python slice values.

        string or list of strings
            Position of the overscan section using an IRAF section
            format (1-based).
        """
        return self._parse_section('overscan_section', 'BIASSEC', pretty)

    @astro_data_descriptor
    def pixel_scale(self):
        """
        Returns the image scale in arcseconds per pixel

        Returns
        -------
        float
            pixel scale
        """

        pass

    @astro_data_descriptor
    def read_mode(self):
        pass

    @returns_list
    @astro_data_descriptor
    def read_noise(self):
        """
        Returns the image scale in arcseconds per pixel

        Returns
        -------
        float
            pixel scale
        """

        pass

    @astro_data_descriptor
    def read_speed_setting(self):
        pass

    @returns_list
    @astro_data_descriptor
    def saturation_level(self):
        """
        Returns the saturation level (in ADU)

        Returns
        -------
        list/float
            saturation level
        """
        pass

    @astro_data_descriptor
    def wcs_ra(self):
        """
        Returns the Right Ascension of the center of the field based on the
        WCS rather than the RA header keyword.

        Returns
        -------
        float
            right ascension in degrees
        """
        pass

    @astro_data_descriptor
    def wcs_dec(self):
        """
        Returns the Declination of the center of the field based on the
        WCS rather than the DEC header keyword.

        Returns
        -------
        float
            declination in degrees
        """
        pass