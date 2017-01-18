import math
import re
from datetime import date

from astrodata import astro_data_tag, astro_data_descriptor, returns_list, TagSet
from .pixel_functions import get_bias_level
from . import lookup
from .. import gmu
from ..gemini import AstroDataGemini


class AstroDataGmos(AstroDataGemini):

    __keyword_dict = dict(array_name = 'AMPNAME',
                          array_section = 'CCDSEC',
                          camera = 'INSTRUME',
                          overscan_section = 'BIASSEC',
                          wavelength_reference_pixel = 'CRPIX1',
                          )

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
        detector_section = self.detector_section(pretty=True)
        # Combine the amp name(s) and detector section(s)
        try:
            read_area = ["'{}':{}".format(a,d) for a,d in zip(ampname, detector_section)]
        except TypeError:
            read_area = "'{}':{}".format(ampname, detector_section)
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
        return self.hdr.AMPNAME

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
                xs *= self.detector_x_bin()
                ys *= self.detector_y_bin()
                roi_list.append((x1, x1+xs-1, y1, y1+ys-1))
            else:
                break
        return roi_list

    @astro_data_descriptor
    def detector_roi_setting(self):
        """
        Looks at the first ROI and returns a descriptive string describing it
        These are more or less the options in the OT

        Returns
        -------
        str
            Name of the ROI setting used or
            "Custom" if the ROI doesn't match
            "Undefined" if there's no ROI in the header
        """
        roi_dict = lookup.gmosRoiSettings
        rois = self.detector_rois_requested()
        if rois:
            roi_setting = 'Custom'
            for s in roi_dict.keys():
                if rois[0] in roi_dict[s]:
                    roi_setting = s
        else:
            roi_setting = 'Undefined'
        return roi_setting

    @astro_data_descriptor
    def detector_x_bin(self):
        """
        Returns the detector binning in the x-direction

        Returns
        -------
        int
            The detector binning
        """
        binning = self.hdr.CCDSUM
        if isinstance(binning, list):
            xbin_list = [b.split()[0] for b in binning]
            # Check list is single-valued
            if xbin_list != xbin_list[::-1]:
                raise ValueError("Multiple values of x-binning!")
            xbin = xbin_list[0]
        else:
            xbin = binning.split()[0]
        return int(xbin)

    @astro_data_descriptor
    def detector_y_bin(self):
        """
        Returns the detector binning in the y-direction

        Returns
        -------
        int
            The detector binning
        """
        binning = self.hdr.CCDSUM
        if isinstance(binning, list):
            ybin_list = [b.split()[1] for b in binning]
            # Check list is single-valued
            if ybin_list != ybin_list[::-1]:
                raise ValueError("Multiple values of y-binning!")
            ybin = ybin_list[0]
        else:
            ybin = binning.split()[1]
        return int(ybin)

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
    def exposure_time(self):
        """
        Returns the exposure time in seconds.

        Returns
        -------
        float
            Exposure time.

        """
        exp_time = float(getattr(self.phu, self._keyword_for('exposure_time')))
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
        mask = self.phu.MASKNAME
        return 'Imaging' if mask=='None' else mask

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

        # Get the correct dict of gain values
        ut_date = self.ut_date()
        if ut_date >= date(2015, 8, 26):
            gain_dict = lookup.gmosampsGain
        elif ut_date >= date(2006, 8, 31):
            gain_dict = lookup.gmosampsGainBefore20150826
        else:
            gain_dict = lookup.gmosampsGainBefore20060831

        read_speed_setting = self.read_speed_setting()
        gain_setting = self.gain_setting()
        # This may be a list
        ampname = self.array_name()

        # Return appropriate object
        if isinstance(ampname, list):
            gain = [gain_dict[read_speed_setting, gain_setting, a]
                    for a in ampname]
        else:
            gain = gain_dict[read_speed_setting, gain_setting, ampname]
        return gain

    @astro_data_descriptor
    def gain_setting(self):
        """
        Returns the gain settings of the extensions. These could be different
        but the old system couldn't handle that so we'll return a string but
        check that they're all the same

        Returns
        -------
        str
            Gain setting
        """
        # This seems to rely on obtaining the original GAIN header keyword
        gain_settings = None
        if 'PREPARED' not in self.tags:
            # Use the (incorrect) GAIN header keywords to determine the setting
            gain = self.hdr.GAIN
        else:
            # For prepared data, we use the value of the gain_setting keyword
            try:
                gain_settings = getattr(self.hdr, self._keyword_for('gain_setting'))
            except KeyError:
                # This code deals with data that haven't been processed with
                # gemini_python (no GAINSET keyword), but have a PREPARED tag
                try:
                    gain = self.hdr.GAINORIG
                    # If GAINORIG is 1 in all the extensions, then the original
                    # gain is actually in GAINMULT(!?)
                    try:
                        if gain == 1:
                            gain = self.hdr.GAINMULT
                    except TypeError:
                        if gain == [1] * len(gain):
                            gain = self.hdr.GAINMULT
                except KeyError:
                    # Use the gain() descriptor as a last resort
                    gain = self.gain()

        # Convert gain to gain_settings if we only got the gain
        if gain_settings is None:
            try:
                gain_settings = ['high' if g > 3.0 else 'low' for g in gain]
            except TypeError:
                gain_settings = 'high' if gain > 3.0 else 'low'

        # Check that all gain settings are the same if multiple extensions
        if isinstance(gain_settings, list):
            if gain_settings != gain_settings[::-1]:
                raise ValueError("Multiple values of gain setting!")
            return gain_settings[0]
        else:
            return gain_settings

    @astro_data_descriptor
    def group_id(self):
        """
        Returns a string representing a group of data that are compatible
        with each other.  This is used when stacking, for example.  Each
        instrument and mode of observation will have its own rules.

        GMOS uses the detector binning, amp_read_area, gain_setting, and
        read_speed_setting. Flats and twilights have the pretty version of
        the filter name included. Science data have the pretty filter name
        and observation_id as well. And spectroscopic data have the grating.
        Got all that?

        Returns
        -------
        str
            A group ID for compatible data.
        """
        tags = self.tags

        # Things needed for all observations
        unique_id_descriptor_list_all = ['detector_x_bin', 'detector_y_bin',
                                             'read_mode', 'amp_read_area']
        if 'SPECT' in tags:
            unique_id_descriptor_list_all.append('disperser')

        # List to format descriptor calls using 'pretty=True' parameter
        call_pretty_version_list = ['filter_name', 'disperser']

        # Force this to be a list
        force_list = ['amp_read_area']

        if 'BIAS' in tags:
            id_descriptor_list = []
        elif 'DARK' in tags:
            id_descriptor_list = ['exposure_time']
        elif 'IMAGE' in tags and ('FLAT' in tags or 'TWILIGHT' in tags):
            id_descriptor_list = ['filter_name']
        else:
            id_descriptor_list = ['observation_id', 'filter_name']

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
            if (descriptor in force_list and
                    not isinstance(descriptor_object, list)):
                descriptor_object = [descriptor_object]

            # Convert descriptor to a string and store
            descriptor_object_string_list.append(str(descriptor_object))

        # Create and return the final group_id string
        return '_'.join(descriptor_object_string_list)

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
        zpt_dict = lookup.nominal_zeropoints

        gain = self.gain()
        filter_name = self.filter_name(pretty=True)
        ccd_name = self.hdr.get('CCDNAME')
        # Explicit: if BUNIT is missing, assume data are in ADU
        bunit = self.hdr.get('BUNIT', 'adu')

        # Have to do the list/not-list stuff here
        # Zeropoints in table are for electrons, so subtract 2.5*log10(gain)
        # if the data are in ADU
        try:
            zpt = [None if ccd is None else
                    zpt_dict[(ccd, filter_name)] -
                   (2.5 * math.log10(g) if b=='adu' else 0)
                   for ccd,g,b in zip(ccd_name, gain, bunit)]
        except TypeError:
            zpt = None if ccd_name is None else \
                zpt_dict[(ccd_name, filter_name)] - (
                2.5 * math.log10(gain) if bunit=='adu' else 0)

        return zpt

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
        Returns the image scale in arcsec per pixel, accounting for binning

        Returns
        -------
        float
            pixel scale
        """
        pixscale_dict = lookup.gmosPixelScales

        # Pixel scale dict is keyed by instrument ('GMOS-N' or 'GMOS-S')
        # and detector type
        pixscale_key = (self.instrument(), self.phu.DETTYPE)
        raw_pixel_scale = pixscale_dict[pixscale_key]
        return raw_pixel_scale * self.detector_y_bin()

    @astro_data_descriptor
    def read_mode(self):
        """
        Returns a string describing the readout mode, which sets the
        gain and readout speed

        Returns
        -------
        str
            read mode used
        """
        # Get the right mapping (detector-dependent)
        det_key = 'Hamamatsu' if \
            self.detector_name(pretty=True)=='Hamamatsu' else 'default'
        mode_dict = lookup.read_mode_map[det_key]
        mode_key = (self.gain_setting(), self.read_speed_setting())
        return mode_dict[mode_key]

    @returns_list
    @astro_data_descriptor
    def read_noise(self):
        """
        Returns the read noise (as a list if multiple extensions, or
        a float if a single-extension slice)

        Returns
        -------
        float/list
            read noise
        """
        if 'PREPARED' in self.tags:
            return self.hdr.get(self._keyword_for('read_noise'))
        else:
            # Get the correct dict of read noise values
            ut_date = self.ut_date()
            if ut_date >= date(2015, 8, 26):
                rn_dict = lookup.gmosampsRdnoise
            elif ut_date >= date(2006, 8, 31):
                rn_dict = lookup.gmosampsRdnoiseBefore20150826
            else:
                rn_dict = lookup.gmosampsRdnoiseBefore20060831

        read_speed_setting = self.read_speed_setting()
        gain_setting = self.gain_setting()
        # This may be a list
        ampname = self.array_name()

        # Return appropriate object
        try:
            read_noise = [rn_dict[read_speed_setting, gain_setting, a]
                    for a in ampname]
        except TypeError:
            read_noise = rn_dict[read_speed_setting, gain_setting, ampname]
        return read_noise

    @astro_data_descriptor
    def read_speed_setting(self):
        """
        Returns the setting for the readout speed (slow or fast)

        Returns
        -------
        str
            the setting for the readout speed
        """
        ampinteg = self.phu.AMPINTEG
        detector = self.detector_name(pretty=True)
        if detector == 'Hamamatsu':
            setting = 'slow' if ampinteg > 8000 else 'fast'
        else:
            setting = 'slow' if ampinteg > 2000 else 'fast'
        return setting

    @astro_data_descriptor
    def saturation_level(self):
        """
        Returns the saturation level (in ADU)

        Returns
        -------
        list/float
            saturation level
        """
        # We need to know whether the data have been bias-subtracted
        # First, look for keywords in PHU
        bias_subtracted = \
            self.phu.get(self._keyword_for('bias_image')) is not None or \
            self.phu.get(self._keyword_for('dark_image')) is not None

        # OVERSCAN keyword also means data have been bias-subtracted
        overscan_levels = self.hdr.get('OVERSCAN')
        # Make bias_subtracted a list if necessary, for each extension
        try:
            bias_subtracted = [bias_subtracted or o is not None
                               for o in overscan_levels]
        except TypeError:
            bias_subtracted |= overscan_levels is not None

        detname = self.detector_name(pretty=True)
        xbin = self.detector_x_bin()
        ybin = self.detector_y_bin()
        bin_factor = xbin * ybin
        ampname = self.array_name()
        gain = self.gain()
        # Explicit: if BUNIT is missing, assume data are in ADU
        bunits = self.hdr.get('BUNIT', 'adu')

        # Get estimated bias levels from LUT
        bias_levels = get_bias_level(self, estimate=True)

        adc_limit = 65535
        # Get the limit that could be processed without hitting the ADC limit
        try:
            # Subtracted bias level if data are bias-substracted, and
            # multiply by gain if data are in electron/electrons
            processed_limit = [(adc_limit - (blev if bsub else 0))
                              * (g if 'electron' in bunit else 1)
                              for blev, bsub, g, bunit in
                              zip(bias_levels, bias_subtracted, gain, bunits)]
        except TypeError:
            processed_limit = (adc_limit - (bias_levels if bias_subtracted
                              else 0)) * (gain if 'electron' in bunits else 1)

        # For old EEV data, or heavily-binned data, we're ADC-limited
        if detname == 'EEV' or bin_factor > 2:
            return processed_limit
        else:
            # Otherwise, we're limited by the electron well depths
            try:
                well_limit = [lookup.gmosThresholds[a] * bin_factor /
                              (g if 'electron' not in bunit else 1) +
                              (blev if not bsub else 0) for a,g,bunit,blev,bsub
                              in zip(ampname, gain, bunits, bias_levels,
                                     bias_subtracted)]
                saturation = [min(w, p) for w, p
                              in zip(well_limit, processed_limit)]
            except TypeError:
                saturation = lookup.gmosThresholds[ampname] * bin_factor / (
                    gain if 'electron' not in bunits else 1) + (bias_levels if
                                                not bias_subtracted else 0)
                if saturation > processed_limit:
                    saturation = processed_limit

        return saturation

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
