import math

from astrodata import astro_data_tag, TagSet, astro_data_descriptor, returns_list
from ..gemini import AstroDataGemini, use_keyword_if_prepared
from .. import gmu
from ..common import build_group_id
from . import lookup

class AstroDataGsaoi(AstroDataGemini):
    __keyword_dict = dict(array_section='CCDSEC',
                          camera='DETECTOR',
                          central_wavelength='WAVELENG',
                          detector_name='DETECTOR',
                          )

    @staticmethod
    def _matches_data(source):
        return source[0].header.get('INSTRUME', '').upper() == 'GSAOI'

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['GSAOI'])

    @astro_data_tag
    def _tag_dark(self):
        if self.phu.get('OBSTYPE') == 'DARK':
            return TagSet(['DARK', 'CAL'], blocks=['IMAGE'])

    @astro_data_tag
    def _tag_image(self):
        tags = ['IMAGE']
        if self.phu.get('OBSTYPE') == 'FLAT':
            tags.extend(['FLAT', 'CAL'])
        if 'DOMEFLAT' in self.phu.get('OBJECT', '').upper():
            tags.extend(['DOMEFLAT', 'FLAT', 'CAL'])
        elif 'TWILIGHT' in self.phu.get('OBJECT', '').upper():
            tags.extend(['TWILIGHT', 'FLAT', 'CAL'])

        return TagSet(tags)

    # Kept separate from _tag_image, because some conditions defined
    # at a higher level conflict with this
    @astro_data_tag
    def _type_gcal_lamp(self):
        obj = self.phu.get('OBJECT', '').upper()
        if obj == 'DOMEFLAT':
            return TagSet(['LAMPON'])
        elif obj == 'DOMEFLAT OFF':
            return TagSet(['LAMPOFF'])

    @returns_list
    @astro_data_descriptor
    def array_name(self):
        """
        Returns a list of the array names of each extension

        Returns
        -------
        list/str
            names of the arrays
        """
        try:
            return self.hdr['ARRAYID']
        except KeyError:
            # Data have been mosaicked, so return the detector name
            # (as a single-element list if necessary)
            return self.phu.get('DETECTOR')

    @astro_data_descriptor
    @gmu.return_requested_units(input_units="AA")
    def central_wavelength(self):
        """
        Returns the central wavelength

        Returns
        -------
        float
            The central wavelength setting
        """
        central_wavelength = self.phu.get('WAVELENG', -1)  # in Angstroms
        if central_wavelength < 0.0:
            return None

        return central_wavelength

    @astro_data_descriptor
    def detector_y_offset(self):
        return -super().detector_y_offset()

    @returns_list
    @use_keyword_if_prepared
    @astro_data_descriptor
    def gain(self):
        """
        Returns the gain (electrons/ADU) of the extensions

        Returns
        -------
        list/float
            gain (e/ADU)
        """
        return self._look_up_arr_property('gain')

    @astro_data_descriptor
    def group_id(self):
        """
        Returns a string representing a group of data that are compatible
        with each other.  This is used when stacking, for example.  Each
        instrument and mode of observation will have its own rules.

        Returns
        -------
        str
            A group ID for compatible data
        """
        # Additional descriptors required for each frame type
        # Note: dark_id and flat_twilight are not in common use.
        #       Those are therefore place holder with initial guess
        #
        #       For flat_id, "on" and "off" domeflats are not taken
        #       with the same obsID, so that association cannot be
        #       made.  The only other sensible characteristic would
        #       be to require a timeframe check, eg. within X hours.
        #
        #       The UT date and local date change in the middle of the
        #       night.  Can't reliably use that.  Thought for a while
        #       using the either the fake UT or equivalently the date
        #       string in the file would work, but found at least one
        #       case where the flat sequence is taken across the 2pm
        #       filename change.
        #
        #       Because the group_id is a static string, I can't use
        #       if-tricks or or-tricks.  The only thing that doesn't
        #       change is the program ID.  That's a bit procedural though
        #       but that's the only thing left.
        #
        #dark_id = ["exposure_time", "coadds"]
        flat_id = ["filter_name", "exposure_time", "program_id"]
        #flat_twilight_id = ["filter_name"]
        science_id = ["observation_id", "filter_name", "exposure_time"]

        # Associate rules with data type
        # Note: add darks and twilight if necessary later.
        if 'FLAT' in self.tags:
            id_descriptor_list = flat_id
        else:
            id_descriptor_list = science_id

        # Add in all the common descriptors required
        id_descriptor_list.extend(["read_mode", "detector_section"])

        return build_group_id(self, id_descriptor_list, prettify=('filter_name'))

    @astro_data_descriptor
    def is_coadds_summed(self):
        """
        Tells whether or not the co-adds have been summed.  If not, they
        have been averaged. GSAOI averages them.

        Returns
        -------
        bool
            True if the data has been summed.  False if it has been averaged.

        """
        return False

    @astro_data_descriptor
    def nominal_photometric_zeropoint(self):
        """
        Returns the nominal zeropoints (i.e., the magnitude corresponding to
        a pixel value of 1) for the extensions in an AD object.
        Zeropoints in table are for electrons, so subtract 2.5*lg(gain)
        if the data are in ADU

        Returns
        -------
        float/list
            zeropoint values, one per SCI extension
        """
        def _zpt(array, filt, gain, in_adu):
            zpt = lookup.nominal_zeropoints.get((filt, array))
            try:
                return zpt - (2.5 * math.log10(gain) if in_adu else 0)
            except TypeError:
                return None

        gain = self.gain()
        filter_name = self.filter_name(pretty=True)
        array_name = self.array_name()
        in_adu = self.is_in_adu()

        if self.is_single:
            return _zpt(array_name, filter_name, gain, in_adu)
        else:
            return [_zpt(a, filter_name, g, in_adu)
                    for a, g in zip(array_name, gain)]

    @use_keyword_if_prepared
    @astro_data_descriptor
    def non_linear_level(self):
        """
        Returns the level at which the data become non-linear, in the units
        of the data.

        Returns
        -------
        int/list
            Value at which the data become non-linear
        """
        # Column 3 gives the fraction of the saturation level at which
        # the data become non-linear
        fraction = self._look_up_arr_property('linlimit')
        sat_level = self.saturation_level()

        if self.is_single:
            try:
                return fraction * sat_level
            except TypeError:
                return None
        else:
            return [f * s if f and s else None
                    for f, s in zip(fraction, sat_level)]

    @returns_list
    @use_keyword_if_prepared
    @astro_data_descriptor
    def read_noise(self):
        """
        Returns the read noise of each extension in electrons, as a float or
        a list of floats

        Returns
        -------
        float/list
            read noise in electrons
        """
        # Column 0 has the read noise (for 1 coadd)
        raw_read_noise = self._look_up_arr_property('readnoise')
        coadd_factor = math.sqrt(self.coadds())
        if self.is_single:
            try:
                return round(raw_read_noise / coadd_factor, 2)
            except TypeError:
                return None
        else:
            return [round(r / coadd_factor, 2) if r else None
                    for r in raw_read_noise]

    @astro_data_descriptor
    def read_speed_setting(self):
        """
        Returns a string describing the read speed setting, as used in the OT

        Returns
        -------
        str
            read speed setting
        """
        # The number of non-destructive reads is the key in the dict
        return lookup.read_modes.get(self.phu.get('LNRS'), 'Unknown')

    @use_keyword_if_prepared
    @astro_data_descriptor
    def saturation_level(self):
        """
        Returns the saturation level in the units of the data for each
        extension, as a list or a single value. Values are obtained from
        a LUT which has the saturation level in ADU, so this is converted
        to electrons using the original gain (from the LUT) and then
        divided by the current gain (from the descriptor), which should
        have been set to 1.0 if the data have been converted to electrons.

        Returns
        -------
        int/list
            saturation level
        """
        welldepth = self._look_up_arr_property('welldepth')
        orig_gain = self._look_up_arr_property('gain')
        gain = self.gain()
        if self.is_single:
            try:
                return welldepth * orig_gain / gain
            except TypeError:
                return None
        return [w * o / g if w and o and g else None for w, o, g in zip(welldepth, orig_gain, gain)]

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
        except KeyError:
            crval = self.phu.get('CRVAL1')
            ctype = self.phu.get('CTYPE1')
        return crval if ctype == 'RA---TAN' else None

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
        except KeyError:
            crval = self.phu.get('CRVAL2')
            ctype = self.phu.get('CTYPE2')

        return crval if ctype == 'DEC--TAN' else None

    def _look_up_arr_property(self, attr):
        """
        Helper function to extract information from the array_properties dict
        Will return a list or a value, depending on the object it's called on

        Returns
        -------
        list/float
            the required data
        """
        read_speed = self.read_speed_setting()
        array_names = self.array_name()

        if isinstance(array_names, list):
            return [getattr(lookup.array_properties.get((read_speed, a)),
                            attr, None) for a in array_names]
        else:
            return getattr(lookup.array_properties.get((read_speed,
                                            array_names)), attr, None)
