import math

from astrodata import astro_data_tag, TagSet, astro_data_descriptor, returns_list
from ..gemini import AstroDataGemini
from .. import gmu
from . import lookup

class AstroDataGsaoi(AstroDataGemini):
    __keyword_dict = dict(array_section='CCDSEC',
                          camera='DETECTOR',
                          central_wavelength='WAVELENG',
                          detector_name='DETECTOR',
                          )

    @staticmethod
    def _matches_data(data_provider):
        return data_provider.phu.get('INSTRUME', '').upper() == 'GSAOI'

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
            return self.hdr.ARRAYID
        except KeyError:
            # Data have been mosaicked, so return the detector name
            # (as a single-element list if necessary)
            return self.phu.DETECTOR

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

        central_wavelength = float(self.phu.WAVELENG)
        if central_wavelength < 0.0:
            raise ValueError("Central wavelength can't be negative!")
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
        return self._look_up_arr_property(1)


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
        # Descriptors used for all image types
        unique_id_descriptor_list_all = ["read_mode", "detector_section"]

        # List to format descriptor calls using 'pretty=True' parameter
        call_pretty_version_list = ["filter_name"]

        # Descriptors to be returned as a list, even if a single extension
        # TODO: Maybe this descriptor should just barf if run on a slice.
        force_list = ["detector_section"]

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

        # non-descriptor strings to attach to the group_id
        #   Note: the local date will be added for flats below.
        additional_item_to_include = None

        # Associate rules with data type
        # Note: add darks and twilight if necessary later.
        tags = self.tags
        if 'FLAT' in tags:
            id_descriptor_list = flat_id
            # get the local date and save as additional item.
            #datestr = re.search('S([0-9]+)S.*', dataset.filename).group(1)
            #additional_item_to_include = [datestr]
        else:
            id_descriptor_list = science_id

        # Add in all the common descriptors required
        id_descriptor_list.extend(unique_id_descriptor_list_all)

        # Form the group_id
        descriptor_object_string_list = []
        for descriptor in id_descriptor_list:
            kw = {}
            if descriptor in call_pretty_version_list:
                kw['pretty'] = True
            descriptor_object = getattr(self, descriptor)(**kw)

            # Ensure we get a list, even if only looking at one extension
            if (descriptor in force_list and
                    not isinstance(descriptor_object, list)):
                descriptor_object = [descriptor_object]

            # Convert descriptor to a string and store
            descriptor_object_string_list.append(str(descriptor_object))

        # Add in any none descriptor related information
        if additional_item_to_include is not None:
            descriptor_object_string_list.append(additional_item_to_include)

        # Create and return the final group_id string
        return '_'.join(descriptor_object_string_list)

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
        zpt_dict = lookup.nominal_zeropoints

        gain = self.gain()
        filter_name = self.filter_name(pretty=True)
        array_name = self.array_name()
        # Explicit: if BUNIT is missing, assume data are in ADU
        bunit = self.hdr.get('BUNIT', 'adu')

        # Have to do the list/not-list stuff here
        # Zeropoints in table are for electrons, so subtract 2.5*log10(gain)
        # if the data are in ADU
        try:
            zpt = [zpt_dict[(filter_name, a)] -
                   (2.5 * math.log10(g) if b=='adu' else 0)
                   for a,g,b in zip(array_name, gain, bunit)]
        except TypeError:
            zpt = zpt_dict[(filter_name, array_name)] - (
                2.5 * math.log10(gain) if bunit=='adu' else 0)

        return zpt


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
        a0 = self._look_up_arr_property(4)
        a1 = self._look_up_arr_property(5)
        a2 = self._look_up_arr_property(6)
        try:
            return [[a, b, c] for a, b, c in zip(a0, a1, a2)]
        except TypeError:
            return [a0, a1, a2]

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
        fraction = self._look_up_arr_property(3)
        saturation_level = self.saturation_level()
        # Saturation level might be an element or a list
        try:
            nonlin_level = [f*s for f, s in zip(fraction,saturation_level)]
        except TypeError:
            nonlin_level = fraction * saturation_level
        return nonlin_level

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
        raw_read_noise = self._look_up_arr_property(0)
        coadd_factor = math.sqrt(self.coadds())
        try:
            return [round(r / coadd_factor, 2) for r in raw_read_noise]
        except TypeError:
            return round(raw_read_noise / coadd_factor, 2)

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
        lnrs = self.phu.LNRS
        try:
            return lookup.read_modes[lnrs]
        except KeyError:
            return 'Invalid'

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
        return self._look_up_arr_property(2)

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
            crval = self.phu.CRVAL1
            ctype = self.phu.CTYPE1

        if ctype == 'RA---TAN':
            return crval
        else:
            raise ValueError('CTYPE1 keyword is not RA---TAN')

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
            crval = self.phu.CRVAL2
            ctype = self.phu.CTYPE2

        if ctype == 'DEC--TAN':
            return crval
        else:
            raise ValueError('CTYPE2 keyword is not DEC--TAN')

    def _look_up_arr_property(self, table_column):
        """
        Helper function to extract information from the array_properties dict
        Will return a list or a value, depending on the object it's called on

        Returns
        -------
        list/float
            the required data
        """
        read_speed_setting = self.read_speed_setting()
        array_names = self.array_name()

        if isinstance(array_names, list):
            return [lookup.array_properties[read_speed_setting, a][table_column]
                    for a in array_names]
        else:
            return lookup.array_properties[read_speed_setting,
                                           array_names][table_column]
