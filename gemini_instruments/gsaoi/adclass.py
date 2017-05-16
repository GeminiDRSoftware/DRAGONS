import math

from astrodata import astro_data_tag, TagSet, astro_data_descriptor, returns_list
from ..gemini import AstroDataGemini
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
    def central_wavelength(self, asMicrometers=False, asNanometers=False,
                           asAngstroms=False):
        """
        Returns the central wavelength in meters or the specified units

        Parameters
        ----------
        asMicrometers : bool
            If True, return the wavelength in microns
        asNanometers : bool
            If True, return the wavelength in nanometers
        asAngstroms : bool
            If True, return the wavelength in Angstroms

        Returns
        -------
        float
            The central wavelength setting

        """
        unit_arg_list = [asMicrometers, asNanometers, asAngstroms]
        if unit_arg_list.count(True) == 1:
            # Just one of the unit arguments was set to True. Return the
            # central wavelength in these units
            if asMicrometers:
                output_units = "micrometers"
            if asNanometers:
                output_units = "nanometers"
            if asAngstroms:
                output_units = "angstroms"
        else:
            # Either none of the unit arguments were set to True or more than
            # one of the unit arguments was set to True. In either case,
            # return the central wavelength in the default units of meters.
            output_units = "meters"

        central_wavelength = self.phu.get('WAVELENG', -1)
        if central_wavelength < 0.0:
            return None
        else:
            return gmu.convert_units('angstroms', central_wavelength,
                                     output_units)

    @returns_list
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

        Returns
        -------
        float/list
            zeropoint values, one per SCI extension
        """
        def _zpt(array, filt, gain, bunit):
            zpt = lookup.nominal_zeropoints.get((filt, array))
            try:
                return zpt - (2.5 * math.log10(gain) if
                              bunit.lower() == 'adu' else 0)
            except TypeError:
                return None

        gain = self.gain()
        filter_name = self.filter_name(pretty=True)
        array_name = self.array_name()
        # Explicit: if BUNIT is missing, assume data are in ADU
        bunit = self.hdr.get('BUNIT', 'adu')

        # Have to do the list/not-list stuff here
        # Zeropoints in table are for electrons, so subtract 2.5*log10(gain)
        # if the data are in ADU
        if self.is_single:
            return _zpt(array_name, filter_name, gain, bunit)
        else:
            return [_zpt(a, filter_name, g, b)
                    for a, g, b, in zip(array_name, gain, bunit)]

    @astro_data_descriptor
    def nonlinearity_coeffs(self):
        """
        For each extension, return a tuple (a0,a1,a2) of coefficients such
        that the linearized counts are a0 + a1*c _ a2*c^2 for raw counts c

        Returns
        -------
        tuple/list
            coefficients
        """
        return self._look_up_arr_property('coeffs')

    @astro_data_descriptor
    def non_linear_level(self):
        """
        Returns the level at which the data become non-linear, in ADU.

        Returns
        -------
        list/float
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

    @astro_data_descriptor
    def read_noise(self):
        """
        Returns the read noise of each extension, as a float or list of floats

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

    @astro_data_descriptor
    def saturation_level(self):
        """
        Returns the saturation level in ADU for each extension, as a list
        or a single value

        Returns
        -------
        float/list
            saturation level in ADU
        """
        return self._look_up_arr_property('welldepth')

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
            crval = h.CRVAL1
            ctype = h.CTYPE1
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
            crval = h.CRVAL2
            ctype = h.CTYPE2
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
