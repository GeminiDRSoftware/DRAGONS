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
        """
        Returns a list of amplifier read areas, one per extension, made by
        combining the amplifier name and detector section. Or returns a
        string if called on a single-extension slice.

        Returns
        -------
        list/str
            read_area of each extension
        """
        ampname = self.array_name()
        detector_section = self.detector_section()
        # Combine the amp name(s) and detector section(s)
        try:
            read_area = ['{}:{}'.format(a,d) for a,d in zip(ampname, detector_section)]
        except TypeError:
            read_area = '{}:{}'.format(ampname, detector_section)
        return read_area

    @astro_data_descriptor
    def array_name(self):
        """
        Returns a list of the names of the arrays of the extensions, or
        a string if called on a single-extension slice

        Returns
        -------
        list/str
            names of the arrays
        """
        return getattr(self.hdr, self._keyword_for('array_name'))

    @astro_data_descriptor
    def central_wavelength(self, asMicrometers=False, asNanometers=False, asAngstroms=False):
        """
        Returns the central wavelength in meters or specified units

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

        central_wavelength = float(self.phu.CENTWAVE)

        if central_wavelength < 0.0:
            raise ValueError("Central wavelength can't be negative!")
        else:
            return gmu.convert_units('nanometers', central_wavelength,
                                     output_units)

    @astro_data_descriptor
    def detector_name(self, pretty=False):
        """
        Returns the name(s) of the detector(s), from the PHU DETID keyword.
        Calling with pretty=True will provide a single descriptive string.

        Parameters
        ----------
        pretty : bool
            If True, return a single descriptive string

        Returns
        -------
        str
            detector name
        """
        if pretty:
            pretty_detname_dict = {
                "SDSU II CCD": "EEV",
                "SDSU II e2v DD CCD42-90": "e2vDD",
                "S10892": "Hamamatsu",
                }
            det_type = self.phu.DETTYPE
            det_name = pretty_detname_dict[det_type]
        else:
            det_name = self.phu.DETID
        return det_name

    @astro_data_descriptor
    def detector_rois_requested(self):
        """
        Returns a list of ROIs, as tuples in a 1-based inclusive (IRAF-like)
        format (x1, x2, y1, y2), in physical (unbinned) pixels.

        Returns
        -------
            list of tuples, one per ROI
        """
        roi_list = []
        for roi in range(1, 10):
            x1 = self.phu.get('DETRO{}X'.format(roi))
            xs = self.phu.get('DETRO{}XS'.format(roi))
            y1 = self.phu.get('DETRO{}Y'.format(roi))
            ys = self.phu.get('DETRO{}YS'.format(roi))
            if x1 is not None:
                xs *= int(self.detector_x_bin())
        return


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
        """
        Returns the name of the grating used for the observation

        Parameters
        ----------
        stripID : bool
            If True, removes the component ID and returns only the name of
            the disperser.
        pretty : bool, also removed the trailing '+'
            If True,
        Returns
        -------
        str
            name of the grating
        """
        stripID |= pretty
        disperser = self.phu.GRATING
        if stripID:
            disperser = gmu.removeComponentID(disperser).strip('+') if pretty \
                else gmu.removeComponentID(disperser)
        return disperser

    @astro_data_descriptor
    def dispersion(self, asMicrometers=False, asNanometers=False, asAngstroms=False):
        """
        Returns the dispersion (wavelength units per pixel) in meters
        or specified units, as a list (one value per extension) or a
        float if used on a single-extension slice.

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
        list/float
            The dispersion(s)
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
        cd11 = self.hdr.CD1_1
        try:
            dispersion = [gmu.convert_units('meters', d, output_units)
                          for d in cd11]
        except TypeError:
            dispersion = gmu.convert_units('meters', cd11, output_units)
        return dispersion

    @astro_data_descriptor
    def dispersion_axis(self):
        """
        Returns the dispersion axis, where 1 is in the x-direction
        and 2 is in the y-direction

        Returns
        -------
        int
            The dispersion axis (1)
        """
        return 1

    @astro_data_descriptor
    def exposure_time(self):
        """
        Returns the exposure time in seconds.

        Returns
        -------
        float
            Exposure time.

        """
        exp_time = float(getattr(self.hdu, self._keyword_for('exposure_time')))
        if exp_time < 0 or exp_time > 10000:
            raise ValueError('Invalid exposure time {}'.format(exp_time))
        return exp_time

    @astro_data_descriptor
    def focal_plane_mask(self, stripID=False, pretty=False):
        """
        Returns the name of the focal plane mask.

        Parameters
        ----------
        stripID : bool
            Doesn't actually do anything.
        pretty : bool
            Same as for stripID

        Returns
        -------
        str
            The name of the focal plane mask
        """
        return getattr(self.phu, self._keyword_for('focal_plane_mask'))

    @returns_list
    @astro_data_descriptor
    def gain(self):
        """
        Returns the gain (electrons/ADU) for each extension

        Returns
        -------
        list/float
            Gains used for the observation

        """
        # If the file has been prepared, we trust the header keywords
        if 'PREPARED' in self.tags:
            return getattr(self.hdr, self._keyword_for('gain'))

        ampinteg = self.phu.AMPINTEG
        ut_date = str(self.ut_date())

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
        """
        Returns a tuple with the number of integrations made in each
        of the nod-and-shuffle positions

        Returns
        -------
        tuple
            number of integrations in the A and B positions
        """
        return (int(self.phu.ANODCNT), int(self.phu.BNODCNT))

    @astro_data_descriptor
    def nod_offsets(self):
        """
        Returns a tuple with the offsets from the default telescope position
        of the A and B nod-and-shuffle positions (in arcseconds)

        Returns
        -------
        tuple
            offsets in arcseconds
        """
        # Cope with two possible situations
        try:
            ayoff = self.phu.NODAYOFF
            byoff = self.phu.NODBYOFF
        except KeyError:
            ayoff = 0.0
            byoff = self.phu.NODYOFF
        return (ayoff, byoff)

    @astro_data_descriptor
    def nod_pixels(self):
        """
        Returns the number of rows that the charge has been shuffled, in
        nod-and-shuffle data

        Returns
        -------
        int
            The number of rows by which the charge is shuffled
        """
        if 'NODANDSHUFFLE' in self.tags:
            return int(self.phu.NODPIX)
        else:
            raise ValueError('nod_pixels() only works for NODANDSHUFFLE data')

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
        For GMOS, this is just the saturation level.

        Returns
        -------
        float/list
            Value(s) at which the data become non-linear
        """
        return self.saturation_level()

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
        WCS rather than the RA keyword. This just uses the CRVAL1 keyword.

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
        WCS rather than the DEC keyword. This just uses the CRVAL2 keyword.

        Returns
        -------
        float
            declination in degrees
        """
        pass