from astrodata import astro_data_tag, TagSet, astro_data_descriptor, returns_list
from ..gemini import AstroDataGemini
import math

from . import lookup
from .. import gmu
from ..common import build_ir_section

class AstroDataNiri(AstroDataGemini):

    # NIRI has no specific keyword overrides

    @staticmethod
    def _matches_data(data_provider):
        return data_provider.phu.get('INSTRUME', '').upper() == 'NIRI'

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['NIRI'])

    @astro_data_tag
    def _tag_dark(self):
        if self.phu.get('OBSTYPE') == 'DARK':
            return TagSet(['DARK', 'CAL'], blocks=['SPECT', 'IMAGE'])

    @astro_data_tag
    def _tag_arc(self):
        if self.phu.get('OBSTYPE') == 'ARC':
            return TagSet(['ARC'])

    @astro_data_tag
    def _tag_flat(self):
        if self.phu.get('OBSTYPE') == 'FLAT':
            return TagSet(['CAL', 'FLAT'])

    @astro_data_tag
    def _tag_image(self):
        if 'grism' not in self.phu.get('FILTER3', ''):
            tags = ['IMAGE']
            if self.phu.get('OBJECT', '').upper() == 'TWILIGHT':
                tags.extend(['CAL', 'FLAT', 'TWILIGHT'])

            return TagSet(tags)

    @astro_data_tag
    def _tag_spect(self):
        if 'grism' in self.phu.get('FILTER3', ''):
            return TagSet(['SPECT', 'LS'])

    @astro_data_descriptor
    def array_section(self, pretty=False):
        """
        Returns the section covered by the array(s) relative to the detector
        frame.  For example, this can be the position of multiple amps read
        within a CCD.  If pretty is False, a tuple of 0-based coordinates
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
            Position of extension(s) using Python slice values

        str/list of str
            Position of extension(s) using an IRAF section format (1-based)
        """
        return build_ir_section(self, pretty)

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

        # Use the lookup dict, keyed on focal_plane_mask and grism
        wave_in_angstroms = lookup.spec_wavlengths.get((self.focal_plane_mask(),
                                                   self.disperser(stripID=True)))
        return gmu.convert_units('angstroms', wave_in_angstroms,
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
        # All NIRI pixels are data
        return self._parse_section('data_section', 'FULLFRAME', pretty)

    @astro_data_descriptor
    def detector_roi_setting(self):
        """
        Returns the ROI setting. Most instruments don't allow this to be
        changed, so at the Gemini level it just returns 'Fixed'

        Returns
        -------
        str
            Name of the ROI setting used, ie, "Fixed"
        """
        data_section = self.data_section()
        # If we have a list, check they're all the same and take one element
        if isinstance(data_section, list):
            assert data_section == data_section[::-1], \
                "Multiple extensions with different data_sections"
            data_section = data_section[0]

        x1, x2, y1, y2 = data_section
        # Check for a sensibly-sized square
        if x1==0 and y1==0 and x2==y2 and x2 % 256==0:
            roi_setting = 'Full Frame' if x2==1024 else 'Central{}'.format(x2)
        else:
            roi_setting = 'Custom'
        return roi_setting

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
    def disperser(self, stripID=False, pretty=False):
        """
        Returns the name of the disperser.  The component ID can be removed
        with either 'stripID' or 'pretty' set to True.

        Parameters
        ----------
        stripID : bool
            If True, removes the component ID and returns only the name of
            the disperser.
        pretty : bool
            Does nothing. Exists for compatibility.

        Returns
        -------
        str
            The name of the disperser with or without the component ID.
        """
        filter3 = self.phu.FILTER3
        if 'grism' in filter3:
            disperser = gmu.removeComponentID(filter3) if stripID else filter3
        else:
            disperser = 'MIRROR'
        return disperser

    @astro_data_descriptor
    def filter_name(self, stripID=False, pretty=False):
        #TODO: Complete rewrite here so serious testing required
        """
        Returns the name of the filter(s) used. If a combination of filters
        is used, the filter names will be join into a unique string with '&' as
        separator. The component IDs can be removed from the filter names.
        Alternatively, a single descriptive filter name can be returned,
        based on a lookup table.

        Parameters
        ----------
        stripID : bool
            If True, removes the component ID(s) and returns only the name(s)
            of the filter(s).
        pretty : bool
            Returns a single filter name

        Returns
        -------
        str
            The name of the filter combination with or without the component ID.
        """
        raw_filters = [self._may_remove_component('FILTER{}'.format(i),
                                        stripID, pretty) for i in [1,2,3]]
        # eliminate any open/grism/pupil from the list
        filters = [f for f in raw_filters if
                   not any(x in f.lower() for x in ['open', 'grism', 'pupil'])]
        filters.sort()

        if 'blank' in filters:
            return 'blank'
        if not filters:
            return 'open'

        if pretty:
            try:
                return lookup.filter_name_mapping[tuple(filters) if len(filters)>1
                                               else filters[0]]
            except KeyError:
                pass
        return '&'.join(filters)

    @returns_list
    @astro_data_descriptor
    def gain(self):
        """
        Returns the gain (electrons/ADU) for each extension

        Returns
        -------
        float/list
            gain
        """
        return lookup.array_properties['gain']

    @astro_data_descriptor
    def group_id(self):
        """
        Returns a string representing a group of data that are compatible
        with each other.  This is used when stacking, for example.  Each
        instrument, mode of observation, and data type will have its own rules.

        Returns
        -------
        str
            A group ID for compatible data.
        """
        tags = self.tags
        if 'DARK' in tags:
            desc_list = ['exposure_time', 'coadds']
        else:
            desc_list = ['observation_id', 'filter_name', 'camera']

        desc_list.extend(['read_mode', 'well_depth_setting', 'detector_section'])
        if 'SPECT' in tags:
            desc_list.extend(['disperser', 'focal_plane_mask'])

        pretty_ones = ['filter_name', 'disperser', 'focal_plane_mask']
        force_list = ['detector_section']

        collected_strings = []
        for desc in desc_list:
            method = getattr(self, desc)
            if desc in pretty_ones:
                result = method(pretty=True)
            else:
                result = method()
            if desc in force_list and not isinstance(result, list):
                result = [result]
            collected_strings.append(str(result))

        if 'IMAGE' in tags and 'FLAT' in tags:
            additional_item = 'NIRI_IMAGE_TWILIGHT' if 'TWILIGHT' in tags \
                else 'NIRI_IMAGE_FLAT'
            collected_strings.append(additional_item)

        return '_'.join(collected_strings)

    @astro_data_descriptor
    def nominal_photometric_zeropoint(self):
        """
        Returns the nominal photometric zeropoint (i.e., magnitude
        corresponding to 1 pixel count) for each extension

        Returns
        -------
        float/list of floats
            Photometric zeropoint
        """
        zpt_dict = lookup.nominal_zeropoints

        gain = self.gain()
        filter_name = self.filter_name(pretty=True)
        camera = self.camera()
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

    @astro_data_descriptor
    def non_linear_level(self):
        """
        Returns the ADU level at which the data become non-linear. A list is
        returned with a value for each extension (i.e., one value for NIRI)
        unless called on a single-extension slice.

        Returns
        -------
        list/int
            non-linearity level in ADU
        """
        saturation_level = self.saturation_level()
        linear_limit = lookup.array_properties['linearlimit']
        try:
            return [int(linear_limit * s) for s in saturation_level]
        except TypeError:
            return int(linear_limit * saturation_level)

    @astro_data_descriptor
    def pupil_mask(self, stripID=False, pretty=False):
        """
        Returns the name of the pupil mask used for the observation

        Returns
        -------
        str
            the pupil mask
        """
        filter3 = self.phu.FILTER3
        if filter3.startswith('pup'):
            pupil_mask = gmu.removeComponentID(filter3) if pretty or stripID \
                else filter3
        else:
            pupil_mask = 'MIRROR'
        return pupil_mask

    @astro_data_descriptor
    def read_mode(self):
        """
        Returns the readout mode used for the observation. This has one of 3
        settings, depending on the number of reads and averages. If these
        numbers do not conform to a standard setting, 'Invalid' is returned

        Returns
        -------
        str
            the read mode used
        """
        setting = (self.phu.LNRS, self.phu.NDAVGS)
        if setting == (16,16):
            read_mode = 'Low Background'
        elif setting == (1,16):
            read_mode = 'Medium Background'
        elif setting == (1,1):
            read_mode = 'High Background'
        else:
            read_mode = 'Invalid'
        return read_mode

    @returns_list
    @astro_data_descriptor
    def read_noise(self):
        """
        Returns the read noise in electrons, as a list unless called on
        a single-extension slice

        Returns
        -------
        float/list
            read noise
        """
        read_mode = self.read_mode()
        if read_mode == 'Low Background':
            key = 'lowreadnoise'
        elif read_mode == 'High Background':
            key = 'readnoise'
        else:
            key = 'medreadnoise'
        read_noise = lookup.array_properties[key]
        # Because coadds are summed, read noise increases by sqrt(COADDS)
        return read_noise * math.sqrt(self.coadds())

    @astro_data_descriptor
    def saturation_level(self):
        """
        Returns the saturation level of the data, in ADU.

        Returns
        -------
        list/float
            saturation level in ADU
        """
        coadds = self.coadds()
        gain = self.gain()
        well = lookup.array_properties[self.well_depth_setting().lower()+'well']
        try:
            saturation_level = [int(well * coadds / g) for g in gain]
        except TypeError:
            saturation_level = int(well * coadds / gain)
        return saturation_level

    @astro_data_descriptor
    def well_depth_setting(self):
        """
        Returns a string describing the well-depth setting of the instrument.
        NIRI has 'Shallow' and 'Deep' options. 'Invalid' is returned if the
        bias voltage doesn't match either setting.

        Returns
        -------
        str
            the well-depth setting
        """
        biasvolt = self.phu.A_VDDUC - self.phu.A_VDET
        if abs(biasvolt - lookup.array_properties['shallowbias']) < 0.05:
            return 'Shallow'
        elif abs(biasvolt - lookup.array_properties['deepbias']) < 0.05:
            return 'Deep'
        return 'Invalid'
