import re
import math

from astrodata import astro_data_tag, TagSet, astro_data_descriptor, returns_list
from ..gemini import AstroDataGemini
from .lookup import array_properties, nominal_zeropoints
from astropy.wcs import WCS, FITSFixedWarning
import warnings

from ..common import section_to_tuple
from .. import gmu

class AstroDataF2(AstroDataGemini):

    __keyword_dict = dict(camera = 'LYOT',
                          central_wavelength = 'GRWLEN',
                          disperser = 'GRISM',
                          focal_plane_mask = 'MOSPOS',
                          )

    @staticmethod
    def _matches_data(data_provider):
        return data_provider.phu.get('INSTRUME', '').upper() in ('F2', 'FLAM')

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['F2'])

    @astro_data_tag
    def _tag_dark(self):
        ot = self.phu.get('OBSTYPE')
        dkflt = False
        for f in (self.phu.get('FILTER1', ''), self.phu.get('FILTER2', '')):
            if re.match('DK.?', f):
                dkflt = True
                break

        if dkflt or ot == 'DARK':
            return TagSet(['DARK', 'CAL'], blocks=['IMAGE', 'SPECT'])

    @astro_data_tag
    def _tag_image(self):
        if self.phu.get('GRISM') == 'Open':
            return TagSet(['IMAGE'])

    # Do not tag this as astro_data_tag. It's a helper function
    def _tag_is_spect(self):
        grism = self.phu.get('GRISM', '')
        grpos = self.phu.get('GRISMPOS', '')

        for pattern in ("JH.?", "HK.?", "R3K.?"):
            if re.match(pattern, grism) or re.match(pattern, grpos):
                return True

        return False

    @astro_data_tag
    def _tag_is_ls(self):
        if not self._tag_is_spect():
            return

        decker = self.phu.get('DECKER') == 'Long_slit' or self.phu.get('DCKERPOS') == 'Long_slit'

        if decker or re.match(".?pix-slit", self.phu.get('MOSPOS', '')):
            return TagSet(['LS', 'SPECT'])

    @astro_data_tag
    def _tag_is_mos(self):
        if not self._tag_is_spect():
            return

        decker = self.phu.get('DECKER') == 'mos' or self.phu.get('DCKERPOS') == 'mos'

        if decker or re.match("mos.?", self.phu.get('MOSPOS', '')):
            return TagSet(['MOS', 'SPECT'])

    @astro_data_tag
    def _tag_arc(self):
        if self.phu.get('OBSTYPE') == 'ARC':
            return TagSet(['ARC', 'CAL'])

    @astro_data_tag
    def _tag_flat(self):
        if self.phu.get('OBSTYPE') == 'FLAT':
            return TagSet(['FLAT', 'CAL'])

    @astro_data_tag
    def _tag_twilight(self):
        if self.phu.get('OBJECT').upper() == 'TWILIGHT':
            rej = set(['FLAT']) if self.phu.get('GRISM') != 'Open' else set()
            return TagSet(['TWILIGHT', 'CAL'], blocks=rej)

    @astro_data_tag
    def _tag_disperser(self):
        disp = self.phu.get('DISPERSR', '')
        if disp.startswith('DISP_WOLLASTON'):
            return TagSet(['POL'])
        elif disp.startswith('DISP_PRISM'):
            return TagSet(['SPECT', 'IFU'])

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

        value_filter = (str if pretty else section_to_tuple)
        # TODO: discover reason why this is hardcoded, rather than from keyword
        return value_filter('[1:2048,1:2048]')

    # TODO: sort out the unit-handling here
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

        central_wavelength = float(self.phu.GRWLEN)
        if self.phu.FILTER1 == 'K-long-G0812':
                central_wavelength = 2.2

        if central_wavelength < 0.0:
            raise ValueError("Central wavelength can't be negative!")
        else:
            return gmu.convert_units('micrometers', central_wavelength,
                                     output_units)

    @astro_data_descriptor
    def data_section(self, pretty=False):
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
        return self.array_section(pretty=pretty)

    @astro_data_descriptor
    def detector_section(self, pretty=False):
        """
        Returns the section covered by the detector relative to the whole
        mosaic of detectors.  If pretty is False, a tuple of 0-based coordinates
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
            Position of the detector using Python slice values.

        string or list of strings
            Position of the detector using an IRAF section format (1-based).

        """
        return self.array_section(pretty=pretty)

    @astro_data_descriptor
    def filter_name(self, stripID=False, pretty=False):
        """
        Returns the name of the filter(s) used.  The component ID can be
        removed with either 'stripID' or 'pretty'.  If a combination of filters
        is used, the filter names will be join into a unique string with '&' as
        separator.  If 'pretty' is True, filter positions such as 'Open',
        'Dark', 'blank', and others are removed leaving only the relevant
        filters in the string.

        Parameters
        ----------
        stripID : bool
            If True, removes the component ID and returns only the name of
            the filter.
        pretty : bool
            Parses the combination of filters to return a single string value
            wi the "effective" filter.

        Returns
        -------
        str
            The name of the filter combination with or without the component ID.

        """
        try:
            filter1 = self.phu.FILTER1
            filter2 = self.phu.FILTER2
        except:
            # Old (pre-20100301) keyword names
            filter1 = self.phu.FILT1POS
            filter2 = self.phu.FILT2POS

        if stripID or pretty:
            filter1 = gmu.removeComponentID(filter1)
            filter2 = gmu.removeComponentID(filter2)

        filter = [filter1, filter2]
        if pretty:
            # Remove filters with the name 'open'
            if 'open' in filter2 or 'Open' in filter2:
                del filter[1]
            if 'open' in filter1 or 'Open' in filter1:
                del filter[0]

            if ('Block' in filter1 or 'Block' in filter2 or 'Dark' in filter1
                or 'Dark' in filter2):
                filter = ['blank']
            if 'DK' in filter1 or 'DK' in filter2:
                filter = ['dark']

            if len(filter) == 0:
                filter = ['open']

        # Return &-concatenated names if we still have two filter names
        return str(filter[0]) if len(filter)==1 else '{}&{}'.format(*filter)

    @returns_list
    @astro_data_descriptor
    def gain(self):
        """
        Returns the gain used for the observation.  This is read from a
        lookup table using the read_mode and the well_depth.

        Returns
        -------
        float
            Gain used for the observation

        """
        lnrs = self.phu.LNRS
        # F2 adds the reads (in ADU), so the electron-to-ADU conversion
        # needs to be divided by the number of reads
        gain = array_properties[lnrs][1] / lnrs
        return gain

    @astro_data_descriptor
    def group_id(self):
        """
        Returns a string representing a group of data that are compatible
        with each other.  This is used when stacking, for example.  Each
        instrument and mode of observation will have its own rules. F2's
        is quite a mouthful.

        Returns
        -------
        str
            A group ID for compatible data.

        """
        # essentially a copy of the NIRI group_id descriptor,
        # adapted for F2.

        # Descriptors used for all image types
        unique_id_descriptor_list_all = ["read_mode", "detector_section"]


        # List to format descriptor calls using 'pretty=True' parameter
        call_pretty_version_list = ["filter_name", "disperser",
                                    "focal_plane_mask"]

        # Descriptors to be returned as a list, even if a single extension
        # TODO: Maybe this descriptor should just barf if run on a slice.
        convert_to_list_list = ["detector_section"]

        # Other descriptors required for spectra
        required_spectra_descriptors = ["disperser", "focal_plane_mask"]
        if "SPECT" in self.tags:
            unique_id_descriptor_list_all.extend(required_spectra_descriptors)

        # Additional descriptors required for each frame type
        dark_id = ["exposure_time", "coadds"]
        flat_id = ["filter_name", "camera", "exposure_time", "observation_id"]
        flat_twilight_id = ["filter_name", "camera"]
        science_id = ["observation_id", "filter_name", "camera", "exposure_time"]
        ## !!! KL: added exposure_time to science_id for QAP.  The sky subtraction
        ## !!! seems unable to deal with difference exposure time circa Sep 2015.
        ## !!! The on-target dither sky-sub falls over completely.
        ## !!! Also, we do not have a fully tested scale by exposure routine.

        # This is used for imaging flats and twilights to distinguish between
        # the two types
        additional_item_to_include = None

        # Update the list of descriptors to be used depending on image type
        ## This requires updating to cover all spectral types
        ## Possible updates to the classification system will make this usable
        ## at the Gemini level
        tags = self.tags
        if "DARK" in tags:
            id_descriptor_list = dark_id
        elif 'IMAGE' in tags and 'FLAT' in tags:
            id_descriptor_list = flat_id
            additional_item_to_include = "F2_IMAGE_FLAT"
        elif 'IMAGE' in tags and 'TWILIGHT' in tags:
            id_descriptor_list = flat_twilight_id
            additional_item_to_include = "F2_IMAGE_TWILIGHT"
        else:
            id_descriptor_list = science_id

        # Add in all of the common descriptors required
        id_descriptor_list.extend(unique_id_descriptor_list_all)

        # Form the group_id
        descriptor_object_string_list = []
        for descriptor in id_descriptor_list:
            kw = {}
            if descriptor in call_pretty_version_list:
                kw['pretty'] = True
            descriptor_object = getattr(self, descriptor)(**kw)

            # Ensure we get a list, even if only looking at one extension
            if (descriptor in convert_to_list_list and
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
    def instrument(self):
        """
        Returns the name of the instrument, coping with the fact that early
        data apparently had the keyword INSTRUME='Flam'

        Returns
        -------
        str
            The name of the instrument, namely 'F2'
        """
        return 'F2'

    # TODO: Don't think this is used. camera() returns the same thing
    @astro_data_descriptor
    def lyot_stop(self, stripID=False, pretty=False):
        """
        Returns the name of the Lyot stop used.  The component ID can be
        removed with either 'stripID' or 'pretty', which do exactly the
        same thing.

        Parameters
        ----------
        stripID : bool
            If True, removes the component ID.
        pretty : bool
            Same as for stripID.

        Returns
        -------
        str
            The name of the Lyot stop with or without the component ID.
        """
        return self._may_remove_component('LYOT', stripID, pretty)

    @returns_list
    @astro_data_descriptor
    def nominal_photometric_zeropoint(self):
        """
        Returns the nominal zeropoints (i.e., the magnitude corresponding to
        a pixel value of 1) for the extensions in an AD object.

        Returns
        -------
        list/float
            zeropoint values, one per SCI extension
        """
        zpt_dict = nominal_zeropoints

        gain = self.gain()
        filter_name = self.filter_name(pretty=True)
        camera = self.camera(pretty=True)
        # Explicit: if BUNIT is missing, assume data are in ADU
        bunit = self.hdr.get('BUNIT', 'adu')

        # Have to do the list/not-list stuff here
        # Zeropoints in table are for electrons, so subtract 2.5*log10(gain)
        # if the data are in ADU
        try:
            zpt = [zpt_dict[filter_name, camera] -
                   (2.5 * math.log10(g) if b=='adu' else 0)
                   for g,b in zip(gain, bunit)]
        except TypeError:
            zpt = zpt_dict[camera, filter_name] - (
                2.5 * math.log10(gain) if bunit=='adu' else 0)

        return zpt

    @returns_list
    @astro_data_descriptor
    def non_linear_level(self):
        """
        Returns the level at which the data become non-linear, in ADU.

        Returns
        -------
        float
            Value at which the data become non-linear
        """
        # Element [3] gives the fraction of the saturation level at which
        # the data become non-linear
        fraction = array_properties[self.phu.LNRS][3]
        saturation_level = self.saturation_level()
        # Saturation level might be an element or a list
        try:
            nonlin_level = [fraction * s for s in saturation_level]
        except TypeError:
            nonlin_level = fraction * saturation_level
        return nonlin_level

    # TODO: is 'F2_DARK' still a tag?
    @astro_data_descriptor
    def observation_type(self):
        """
        Returns the observation type (OBJECT, DARK, BIAS, etc.)

        Returns
        -------
        str
            Observation type
        """
        return 'DARK' if 'F2_DARK' in self.tags else self.phu.OBSTYPE

    @astro_data_descriptor
    def pixel_scale(self):
        """
        Returns the image scale in arcseconds per pixel

        Returns
        -------
        float
            pixel scale
        """
        # Try to use the Gemini-level helper method
        try:
            pixel_scale = self._get_wcs_pixel_scale()
        except KeyError:
            pixel_scale = self.phu.PIXSCALE
        return pixel_scale

    @astro_data_descriptor
    def read_mode(self):
        """
        Returns the read mode (i.e., the number of non-destructive read pairs)

        Returns
        -------
        str
            readout mode
        """
        return str(self.phu.LNRS)

    @returns_list
    @astro_data_descriptor
    def read_noise(self):
        """
        Returns the read noise in electrons

        Returns
        -------
        float
            read noise
        """
        # Element [0] gives the read noise
        return array_properties[self.phu.LNRS][0]

    @returns_list
    @astro_data_descriptor
    def saturation_level(self):
        """
        Returns the saturation level (in ADU)

        Returns
        -------
        float
            saturation level
        """
        # Element [2] gives the saturation level in electrons
        saturation_electrons = array_properties[self.phu.LNRS][2]
        gain = self.gain()
        # Gain might be an element or a list
        try:
            saturation_adu = [saturation_electrons / g for g in gain]
        except TypeError:
            saturation_adu = saturation_electrons / gain
        return saturation_adu

    # TODO: document why these are reversed
    @astro_data_descriptor
    def x_offset(self):
        """
        Returns the x offset from origin of this image

        Returns
        -------
        float
            x offset
        """
        return -self.phu.YOFFSET

    @astro_data_descriptor
    def y_offset(self):
        """
        Returns the y offset from origin of this image

        Returns
        -------
        float
            y offset
        """
        return -self.phu.XOFFSET

    def _get_wcs_coords(self):
        """
        Returns the RA and dec of the middle of the data array

        Returns
        -------
        tuple
            (right ascension, declination)
        """
        # Cass rotator centre (according to Andy Stephens from gacq)
        x, y = 1034, 1054
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FITSFixedWarning)
            # header[0] is PHU, header[1] is first (and only) extension HDU
            wcs = WCS(self.header[1])
            result = wcs.wcs_pix2world(x,y,1, 1) if wcs.naxis==3 \
                else wcs.wcs_pix2world(x,y, 1)
        ra, dec = float(result[0]), float(result[1])

        if 'NON_SIDEREAL' in self.tags:
            ra, dec = gmu.toicrs('APPT', ra, dec, ut_datetime=self.ut_datetime())

        return (ra, dec)
