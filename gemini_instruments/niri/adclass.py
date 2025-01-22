from astrodata import astro_data_tag, TagSet, astro_data_descriptor, returns_list
from ..gemini import AstroDataGemini, use_keyword_if_prepared
import math
import re

from . import lookup
from .. import gmu
from ..common import build_ir_section, build_group_id

class AstroDataNiri(AstroDataGemini):

    # NIRI has no specific keyword overrides

    @staticmethod
    def _matches_data(source):
        return source[0].header.get('INSTRUME', '').upper() == 'NIRI'

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
    @gmu.return_requested_units()
    def central_wavelength(self):
        """
        Returns the central wavelength in nm

        Returns
        -------
        float
            The central wavelength setting in nm
        """
        # Use the lookup dict, keyed on camera, focal_plane_mask and grism
        camera = self.camera()
        try:
            disperser = self.disperser(stripID=True)[0:6]
        except TypeError:
            disperser = None
        fpmask = self.focal_plane_mask(stripID=True)

        try:
            return lookup.spec_wavelengths[camera, fpmask, disperser].cenwave
        except KeyError:
            return None

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
        return self._parse_section('FULLFRAME', pretty)

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
        elif data_section is None:
            return None

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
    def detector_x_offset(self):
        """
        Returns the offset from the reference position in pixels along
        the positive x-direction of the detector

        Returns
        -------
        float
            The offset in pixels
        """
        try:
            offset = self.phu.get('POFFSET') / self.pixel_scale()
        except TypeError:  # either is None
            return None
        # Flipped if on bottom port unless AO is operating
        return -offset if (self.phu.get('INPORT')==1 and
                           not self.is_ao()) else offset
    @astro_data_descriptor
    def detector_y_offset(self):
        """
        Returns the offset from the reference position in pixels along
        the positive y-direction of the detector

        Returns
        -------
        float
            The offset in pixels
        """
        try:
            return -self.phu.get('QOFFSET') / self.pixel_scale()
        except TypeError:  # either is None
            return None

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
        stripID |= pretty
        try:
            filter3 = self.phu['FILTER3']
        except KeyError:
            return None
        if 'grism' in filter3:
            return gmu.removeComponentID(filter3) if stripID else filter3
        else:
            return 'MIRROR'

    @astro_data_descriptor
    @gmu.return_requested_units()
    def dispersion(self):
        """
        Returns the dispersion in nm per pixel as a list (one value per
        extension) or a float if used on a single-extension slice. It is
        possible to control the units of wavelength using the input arguments.

        Returns
        -------
        list/float
            The dispersion(s)
        """
        camera = self.camera()
        try:
            disperser = self.disperser(stripID=True)[0:6]
        except TypeError:
            disperser = None

        try:
            dispersion = lookup.dispersion_by_config[camera, disperser]
        except KeyError:
            dispersion = None

        if dispersion is not None and not self.is_single:
                dispersion = [dispersion] * len(self)

        return dispersion

    @returns_list
    @astro_data_descriptor
    def dispersion_axis(self):
        return 1

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
        filters = [f for f in raw_filters if f is not None and
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
    @use_keyword_if_prepared
    @astro_data_descriptor
    def gain(self):
        """
        Returns the gain (electrons/ADU) for each extension

        Returns
        -------
        float/list
            gain
        """
        return lookup.array_properties.get('gain')

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

        desc_list.extend(['read_mode', 'well_depth_setting',
                          'detector_section'])
        if 'SPECT' in tags:
            desc_list.extend(['disperser', 'focal_plane_mask'])

        if 'IMAGE' in tags and 'FLAT' in tags:
            additional_item = 'NIRI_IMAGE_TWILIGHT' if 'TWILIGHT' in tags \
                else 'NIRI_IMAGE_FLAT'
        else:
            additional_item = None

        return build_group_id(self, desc_list, prettify=['filter_name',
                              'disperser', 'focal_plane_mask'],
                              additional=additional_item)

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
        gain = self.gain()
        filter_name = self.filter_name(pretty=True)
        camera = self.camera()
        in_adu = self.is_in_adu()
        zpt = lookup.nominal_zeropoints.get((filter_name, camera))

        # Zeropoints in table are for electrons, so subtract 2.5*log10(gain)
        # if the data are in ADU
        if self.is_single:
            try:
                return zpt - (2.5 * math.log10(gain) if in_adu else 0)
            except TypeError:
                return None
        else:
            return [zpt - (2.5 * math.log10(g) if in_adu else 0) if zpt and g
                    else None for g in gain]

    @use_keyword_if_prepared
    @astro_data_descriptor
    def non_linear_level(self):
        """
        Returns the level at which the data become non-linear, in units of the
        data.

        Returns
        -------
        int/list
            non-linearity level
        """
        sat_level = self.saturation_level()
        linear_limit = lookup.array_properties['linearlimit']
        if isinstance(sat_level, list):
            return [(int(linear_limit * s) if s else None) for s in sat_level]
        else:
            return int(linear_limit * sat_level) if sat_level else None

    @astro_data_descriptor
    def pupil_mask(self, stripID=False, pretty=False):
        """
        Returns the name of the pupil mask used for the observation

        Returns
        -------
        str
            the pupil mask
        """
        try:
            filter3 = self.phu['FILTER3']
        except KeyError:
            return None
        if filter3.startswith('pup'):
             return gmu.removeComponentID(filter3) if pretty or stripID \
                else filter3
        else:
            return 'MIRROR'

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
        setting = (self.phu.get('LNRS'), self.phu.get('NDAVGS'))
        if setting == (16,16):
            return 'Low Background'
        elif setting == (1,16):
            return 'Medium Background'
        elif setting == (1,1):
            return 'High Background'
        else:
            return 'Unknown'

    @returns_list
    @use_keyword_if_prepared
    @astro_data_descriptor
    def read_noise(self):
        """
        Returns the read noise in electrons, as a list unless called on
        a single-extension slice.

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
        try:
            read_noise = lookup.array_properties[key]
        except KeyError:
            return None
        # Because coadds are summed, read noise increases by sqrt(COADDS)
        return read_noise * math.sqrt(self.coadds())

    @use_keyword_if_prepared
    @astro_data_descriptor
    def saturation_level(self):
        """
        Returns the saturation level of the data, in the units of the data

        Returns
        -------
        int/list
            saturation level
        """
        coadds = self.coadds()
        gain = self.gain()
        well = lookup.array_properties.get(self.well_depth_setting().lower()+'well')
        if self.is_single:
            try:
                return int(well * coadds / gain)
            except TypeError:
                return None
        else:
            return [int(well * coadds / g) if g and well else None for g in gain]

    @astro_data_descriptor
    def slit_width(self):
        """
        Returns the width of the slit in arcseconds

        Returns
        -------
        float/None
            the slit width in arcseconds
        """
        fpmask = self.focal_plane_mask(pretty=True)
        if 'pix' in fpmask:
            m = re.match('f(.*)-(.*)pix', fpmask)
            return int(m.group(2)) * 0.7 / int(m.group(1))
        return None

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
        try:
            biasvolt = self.phu['A_VDDUC'] - self.phu['A_VDET']
            if abs(biasvolt - lookup.array_properties['shallowbias']) < 0.05:
                return 'Shallow'
            elif abs(biasvolt - lookup.array_properties['deepbias']) < 0.05:
                return 'Deep'
        except KeyError:
            pass
        return 'Unknown'

    @gmu.return_requested_units(input_units="nm")
    def actual_central_wavelength(self):
        camera = self.camera()
        try:
            disperser = self.disperser(stripID=True)[0:6]
        except TypeError:
            disperser = None
        fpmask = self.focal_plane_mask(stripID=True)
        try:
            return lookup.spec_wavelengths[camera, fpmask, disperser].cenpixwave
        except KeyError:
            return None
