import re
import os
import math
import datetime
import dateutil.parser
import warnings

from astropy.wcs import WCS, FITSFixedWarning
from astrodata import AstroDataFits, astro_data_tag, astro_data_descriptor, TagSet
from astrodata import factory, simple_descriptor_mapping, keyword
from .lookup import wavelength_band, nominal_extinction, filter_wavelengths

# NOTE: Temporary functions for test. gempy imports astrodata and
#       won't work with this implementation
from ..gmu import *

# TODO: Some of these should go to AstroDataFITS
gemini_keyword_names = dict(
    airmass = 'AIRMASS',
    ao_fold = 'AOFOLD',
    ao_seeing = 'AOSEEING',
    array_name = 'ARRAYNAM',
    array_section = 'ARRAYSEC',
    azimuth = 'AZIMUTH',
    bias_image = 'BIASIM',
    bunit = 'BUNIT',
    camera = 'CAMERA',
    cass_rotator_pa = 'CRPA',
    cd11 = 'CD1_1',
    cd12 = 'CD1_2',
    cd21 = 'CD2_1',
    cd22 = 'CD2_2',
    central_wavelength = 'CWAVE',
    coadds = 'COADDS',
    dark_image = 'DARKIM',
    data_label = 'DATALAB',
    data_section = 'DATASEC',
    dec = 'DEC',
    decker = 'DECKER',
    detector_name = 'DETNAME',
    detector_roi_setting = 'DROISET',
    detector_rois_requested = 'DROIREQ',
    detector_section = 'DETSEC',
    detector_x_bin = 'XCCDBIN',
    detector_y_bin = 'YCCDBIN',
    disperser = 'DISPERSR',
    dispersion = 'WDELTA',
    dispersion_axis = 'DISPAXIS',
    elevation = 'ELEVATIO',
    exposure_time = 'EXPTIME',
    filter_name = 'FILTNAME',
    focal_plane_mask = 'FPMASK',
    gain = 'GAIN',
    gain_setting = 'GAINSET',
    gems = 'GWFS1CFG',
    grating = 'GRATING',
    group_id = 'GROUPID',
    local_time = 'LT',
    lyot_stop = 'LYOTSTOP',
    mdf_row_id = 'MDFROW',
    naxis1 = 'NAXIS1',
    naxis2 = 'NAXIS2',
    nominal_atmospheric_extinction = 'NOMATMOS',
    nominal_photometric_zeropoint = 'NOMPHOTZ',
    non_linear_level = 'NONLINEA',
    observation_class = 'OBSCLASS',
    observation_epoch = 'OBSEPOCH',
    observation_id = 'OBSID',
    observation_type = 'OBSTYPE',
    oiwfs = 'OIWFS_ST',
    overscan_section = 'OVERSSEC',
    pixel_scale = 'PIXSCALE',
    prism = 'PRISM',
    program_id = 'GEMPRGID',
    pupil_mask = 'PUPILMSK',
    pwfs1 = 'PWFS1_ST',
    pwfs2 = 'PWFS2_ST',
    qa_state = 'QASTATE',
    r_zero_val = 'RZEROVAL',
    ra = 'RA',
    raw_central_wavelength = 'CWAVE',
    raw_gemini_qa = 'RAWGEMQA',
    raw_pi_requirements_met = 'RAWPIREQ',
    read_mode = 'READMODE',
    read_noise = 'RDNOISE',
    read_speed_setting = 'RDSPDSET',
    saturation_level = 'SATLEVEL',
    slit = 'SLIT',
    ut_datetime = 'DATETIME',
    ut_time = 'UT',
    wavefront_sensor = 'WFS',
    wavelength = 'WAVELENG',
    wavelength_band = 'WAVEBAND',
    wavelength_reference_pixel = 'WREFPIX',
    well_depth_setting = 'WELDEPTH',
    x_offset = 'XOFFSET',
    y_offset = 'YOFFSET',
)

# These construct descriptors that simply get that keyword from the PHU
# These will override any descriptor functions in the class definition
gemini_simple_descriptors = dict(
    azimuth = keyword('AZIMUTH'),
    data_label = keyword('DATALAB'),
    dispersion = keyword('WDELTA'),
    elevation = keyword('ELEVATIO'),
    grating = keyword('GRATING'),
    group_id = keyword('GROUPID'),
    lyot_stop = keyword('LYOTSTOP'),
    nominal_photometric_zeropoint = keyword('NOMPHOTZ'),
    observation_class = keyword('OBSCLASS'),
    observation_epoch = keyword('OBSEPOCH'),
    observation_id = keyword('OBSID'),
    observation_type = keyword('OBSTYPE'),
    prism = keyword('PRISM'),
    program_id = keyword('GEMPRGID'),
    pupil_mask = keyword('PUPILMSK'),
    r_zero_val = keyword('RZEROVAL'),
    slit = keyword('SLIT'),
    wavelength = keyword('WAVELENG'),
    x_offset = keyword('XOFFSET'),
    y_offset = keyword('YOFFSET'),
)

@simple_descriptor_mapping(**gemini_simple_descriptors)
class AstroDataGemini(AstroDataFits):
    __keyword_dict = gemini_keyword_names

    @staticmethod
    def _matches_data(data_provider):
        obs = data_provider.header[0].get('OBSERVAT', '').upper()
        # This covers variants like 'Gemini-North', 'Gemini North', etc.
        return obs in ('GEMINI-NORTH', 'GEMINI-SOUTH')

    @astro_data_tag
    def _type_observatory(self):
        return TagSet(['GEMINI'])

    @astro_data_tag
    def _type_acquisition(self):
        if self.phu.OBSCLASS in ('acq', 'acqCal'):
            return TagSet(['ACQUISITION'])

    @astro_data_tag
    def _type_az(self):
        if self.phu.FRAME == 'AZEL_TOPO':
            try:
                if self.phu.get('ELEVATIO', 0) >= 90:
                    return TagSet(['AZEL_TARGET', 'AT_ZENITH'])
            except ValueError:
                pass
            return TagSet(['AZEL_TARGET'])

    @astro_data_tag
    def _type_fringe(self):
        if self.phu.GIFRINGE is not None:
            return TagSet(['FRINGE'])

    # GCALFLAT and the LAMPON/LAMPOFF are kept separated because the
    # PROCESSED status will cancel the tags for lamp status, but the
    # GCALFLAT is still needed
    @astro_data_tag
    def _type_gcalflat(self):
        if self.phu.GCALLAMP == 'IRhigh':
            return TagSet(['GCALFLAT', 'FLAT', 'CAL'])

    @astro_data_tag
    def _type_gcal_lamp(self):
        if self.phu.GCALLAMP == 'IRhigh':
            shut = self.phu.GCALSHUT
            if shut == 'OPEN':
                return TagSet(['GCAL_IR_ON', 'LAMPON'], blocked_by=['PROCESSED'])
            elif shut == 'CLOSED':
                return TagSet(['GCAL_IR_OFF', 'LAMPOFF'], blocked_by=['PROCESSED'])

    @astro_data_tag
    def _type_site(self):
        site = self.phu.get('OBSERVAT', '').upper()

        if site == 'GEMINI-NORTH':
            return TagSet(['NORTH'])
        elif site == 'GEMINI-SOUTH':
            return TagSet(['SOUTH'])

    @astro_data_tag
    def _type_nodandchop(self):
        if self.phu.DATATYPE == "marked-nodandchop":
            return TagSet(['NODCHOP'])

    @astro_data_tag
    def _type_sidereal(self):
        frames = set([self.phu.get('TRKFRAME'), self.phu.get('FRAME')])
        valid_frames = set(['FK5', 'APPT'])

        # Check if the intersection of both sets is non-empty...
        if frames & valid_frames:
            try:
                dectrack, ratrack = float(self.phu.DECTRACK), float(self.phu.RATRACK)
                if dectrack == 0 and ratrack == 0:
                    return TagSet(['SIDEREAL'])
            except (ValueError, TypeError, KeyError):
                pass
            return TagSet(['NON_SIDEREAL'])

    @astro_data_tag
    def _status_raw(self):
        if 'GEM-TLM' not in self.phu:
            return TagSet(['RAW'])

    @astro_data_tag
    def _status_prepared(self):
        if any(('PREPAR' in kw) for kw in self.phu.keywords):
            return TagSet(['PREPARED'])
        else:
            return TagSet(['UNPREPARED'])

    @astro_data_tag
    def _status_overscan(self):
        found = []
        for pattern, tag in (('TRIMOVER', 'OVERSCAN_TRIMMED'), ('SUBOVER', 'OVERSCAN_SUBTRACTED')):
            if any((pattern in kw) for kw in self.phu.keywords):
                found.append(tag)
        if found:
            return TagSet(found)

    @astro_data_tag
    def _status_processed_cals(self):
        kwords = set(['PROCARC', 'GBIAS', 'PROCBIAS', 'PROCDARK',
                      'GIFLAG', 'PROCFLAT', 'GIFRINGE', 'PROCFRNG'])

        if set(self.phu.keywords) & kwords:
            return TagSet(['PROCESSED'])

    @astro_data_tag
    def _status_processed_science(self):
        for pattern in ('GMOSAIC', 'PREPAR'):
            if not any((pattern in kw) for kw in self.phu.keywords):
                return

        if self.phu.OBSTYPE == 'OBJECT':
            return TagSet(['PROCESSED_SCIENCE'])

    def _parse_section(self, descriptor_name, keyword, pretty):
        try:
            value_filter = (str if pretty else sectionStrToIntList)
            process_fn = lambda x: (None if x is None else value_filter(x))
            sections = self.hdr.get(keyword)
            if self._single:
                return process_fn(sections)
            else:
                return [process_fn(raw) for raw in sections]
        except KeyError:
            raise AttributeError("No {} information".format(descriptor_name))

    def _may_remove_component(self, keyword, stripID, pretty):
        val = self.phu.get(keyword)
        if stripID or pretty:
            return removeComponentID(val)
        return val

    @property
    def instrument_name(self):
        return self.instrument().upper()

    @astro_data_descriptor
    def airmass(self):
        """
        Returns the airmass of the observation.

        Returns
        -------
        float
            Airmass value.

        """
        am = float(getattr(self.phu, self._keyword_for('airmass')))

        if am < 1:
            raise ValueError("Can't have less than 1 airmass!")

        return am

    @astro_data_descriptor
    def ao_seeing(self):
        """
        Returns an estimate of the natural seeing as calculated from the
        adaptive optics systems.

        Returns
        -------
        float
            AO estimate of the natural seeing

        """
        try:
            return self.phu.AOSEEING
        except KeyError:
            try:
                # If r_zero_val (Fried's parameter) is present, 
                # a seeing estimate can be calculated (NOTE: Jo Thomas-Osip 
                # is providing a reference for this calculation. Until then, 
                # EJD checked using 
                # http://www.ctio.noao.edu/~atokovin/tutorial/part1/turb.html )

                # Seeing at 0.5 micron
                rzv = getattr(self.phu, self._keyword_for('r_zero_val'))
                return (206265. * 0.98 * 0.5e-6) / (rzv * 0.01)
            except KeyError:
                raise AttributeError("There is no information about AO seeing")

    # TODO: Clean up the array_section output interface. Trac #821
    @astro_data_descriptor
    def array_section(self, pretty=False):
        """
        Returns the section covered by the array(s) relative to the detector
        frame.  For example, this can be the position of multiple amps read
        within a CCD.  If pretty is False, a tuple of 0-based coordinates
        is returned with format (x1, x2, y1, y2).  If pretty is True, a keyword
        value is returned without parsing as a string.  In this format, the
        coordinates are generally 1-based.

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
            Position of extension(s) using Python slice values

        string or list of strings
            Position of extension(s) using an IRAF section format (1-based)


        """
        return self._parse_section('array_section',
                                   self._keyword_for('array_section'), pretty)

    @astro_data_descriptor
    def camera(self, stripID=False, pretty=False):
        """
        Returns the name of the camera.  The component ID can be removed
        with either 'stripID' or 'pretty' set to True.

        Parameters
        ----------
        stripID : bool
            If True, removes the component ID and returns only the name of
            the camera.
        pretty : bool
            Same as for stripID.  Pretty here does not do anything more.

        Returns
        -------
        str
            The name of the camera with or without the component ID.

        """
        return self._may_remove_component(self._keyword_for('camera'),
                                          stripID, pretty)

    @astro_data_descriptor
    def cass_rotator_pa(self):
        """
        Returns the position angle of the Cassegrain rotator.

        Returns
        -------
        float
            Position angle of the Cassegrain rotator.

        """
        val = float(self.phu.get(self._keyword_for('cass_rotator_pa')))
        if val < -360 or val > 360:
            raise ValueError("Invalid CRPA value: {}".format(val))
        return val

    # TODO: Allow for unit conversion
    @astro_data_descriptor
    def central_wavelength(self):
        """
        Returns the central wavelength in meters.

        Returns
        -------
        float
            The central wavelength setting in meters.

        """
        val = self.raw_central_wavelength()
        if val < 0:
            raise ValueError("Invalid CWAVE value: {}".format(val))

        # We assume that raw_central_wavelength returns micrometers.
        return val / 1e-6

    @astro_data_descriptor
    def coadds(self):
        """
        Returns the number of co-adds used for the observation.

        Returns
        -------
        int
            Number of co-adds.

        """
        return int(self.phu.get(self._keyword_for('coadds'), 1))

    @astro_data_descriptor
    def data_section(self, pretty=False):
        """
        Returns the rectangular section that includes the pixels that would be
        exposed to light.  If pretty is False, a tuple of 0-based coordinates
        is returned with format (x1, x2, y1, y2).  If pretty is True, a keyword
        value is returned without parsing as a string.  In this format, the
        coordinates are generally 1-based.

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
            Location of the pixels exposed to light using Python slice values.

        string or list of strings
            Location of the pixels exposed to light using an IRAF section
            format (1-based).


        """

        return self._parse_section('data_section',
                                   self._keyword_for('data_section'), pretty)

    @astro_data_descriptor
    def decker(self, stripID=False, pretty=False):
        """
        Returns the name of the decker.  The component ID can be removed
        with either 'stripID' or 'pretty' set to True.

        Parameters
        ----------
        stripID : bool
            If True, removes the component ID and returns only the name of
            the decker.
        pretty : bool
            Same as for stripID.  Pretty here does not do anything more.

        Returns
        -------
        str
            The name of the decker with or without the component ID.

        """

        return self._may_remove_component(self._keyword_for('decker'),
                                          stripID, pretty)

    @astro_data_descriptor
    def detector_section(self, pretty=False):
        """
        Returns the section covered by the detector relative to the whole
        mosaic of detectors.  If pretty is False, a tuple of 0-based coordinates
        is returned with format (x1, x2, y1, y2).  If pretty is True, a keyword
        value is returned without parsing as a string.  In this format, the
        coordinates are generally 1-based.

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
            Position of the detector using Python slice values.

        string or list of strings
            Position of the detector using an IRAF section format (1-based).


        """

        return self._parse_section('detector_section',
                                   self._keyword_for('detector_section'), pretty)

    @astro_data_descriptor
    def detector_x_bin(self):
        """
        Returns the detector binning in the x-direction

        Returns
        -------
        int
            The detector binning
        """
        return self.phu.get(self._keyword_for('detector_x_bin', 1))

    @astro_data_descriptor
    def detector_y_bin(self):
        """
        Returns the detector binning in the y-direction

        Returns
        -------
        int
            The detector binning
        """
        return self.phu.get(self._keyword_for('detector_y_bin', 1))

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
            Same as for stripID.  Pretty here does not do anything more.

        Returns
        -------
        str
            The name of the disperser with or without the component ID.

        """

        return self._may_remove_component(self._keyword_for('disperser'),
                                          stripID, pretty)

    @astro_data_descriptor
    def dispersion_axis(self):
        """
        Returns the axis along which the light is dispersed.

        Returns
        -------
        int
            Dispersion axis.

        Raises
        ------
        ValueError
            If the data is tagged IMAGE or is not PREPARED.

        """
        tags = self.tags
        if 'IMAGE' in tags or 'PREPARED' not in tags:
            raise ValueError("This descriptor doesn't work on RAW or IMAGE files")

        # TODO: We may need to sort out Nones here...
        dispaxis_kwd = self._keyword_for('dispersion_axis')
        return [int(dispaxis) for dispaxis in self.hdr.dispaxis_kwd]

    @astro_data_descriptor
    def effective_wavelength(self, output_units=None):
        """
        Returns the wavelength representing the bandpass or the spectrum.
        For imaging data this normally is the wavelength at the center of
        the bandpass as defined by the filter used.  For spectra, this is
        the central wavelength.  The returned value is in meters.

        This descriptor makes uses of a lookup table to associate filters
        with their effective_wavelength.

        Returns
        -------
        float
            Wavelength representing the bandpass or the spectrum coverage.

        """
        if not output_units in ('micrometers', 'nanometers', 'angstroms'):
            output_units = 'meters'
        tags = self.tags
        if 'IMAGE' in tags:
            wave_in_microns = None
            filter_name = self.filter_name(pretty=True)
            for inst in (self.instrument(), '*'):
                try:
                    wave_in_microns = filter_wavelengths[inst, filter_name]
                except KeyError:
                    pass
            if wave_in_microns is None:
                raise KeyError("Can't find the wavelength for this filter in the look-up table")
        elif 'SPECT' in tags:
            wave_in_microns = self.central_wavelength(asMicrometers=True)

        return convert_units('micrometers', wave_in_microns, output_units)

    @astro_data_descriptor
    def exposure_time(self):
        """
        Returns the exposure time in seconds.

        Returns
        -------
        float
            Exposure time.

        """
        exposure_time = getattr(self.phu, self._keyword_for('exposure_time'))
        if exposure_time < 0:
            raise ValueError("Invalid exposure time: {}".format(exposure_time))

        if 'PREPARED' in self.tags and self.is_coadds_summed():
            return exposure_time * self.coadds()
        else:
            return exposure_time

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
            Same as for stripID.  Pretty here does not do anything more.

        Returns
        -------
        str
            The name of the filter combination with or without the component ID.

        """
        f1 = self.phu.FILTER1
        f2 = self.phu.FILTER2

        if stripID or pretty:
            f1 = removeComponentID(f1)
            f2 = removeComponentID(f2)

        if pretty:
            filter_comps = []
            for fn in (f1, f2):
                if "open" not in fn.lower() and "Clear" not in fn:
                    filter_comps.append(fn)
            if not filter_comps:
                filter_comps.append("open")
            cals = (("Block", "blank"), ("Dark", "blank"), ("DK", "dark"))
            for cal, fn in cals:
                if cal in f1 or cal in f2:
                    filter_comps.append(fn)
        else:
            filter_comps = [f1, f2]

        return "&".join(filter_comps[:2])

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
        return self._may_remove_component(self._keyword_for('focal_plane_mask'),
                                   stripID, pretty)

    @astro_data_descriptor
    def gain(self):
        """
        Returns the gain (electrons/ADU) for each extension

        Returns
        -------
        list
            Gains used for the observation

        """
        return self.hdr.get(self._keyword_for('gain'))

    @astro_data_descriptor
    def gcal_lamp(self):
        """
        Returns the name of the GCAL lamp being used, or "Off" if no lamp is
        in used.

        Returns
        -------
        str
            Name of the GCAL lamp being used, or "Off" if not in use.

        """
        try:
            lamps, shut = self.phu.GCALLAMP, self.phu.GCALSHUT
            if (shut.upper() == 'CLOSED' and lamps.upper() in ('IRHIGH', 'IRLOW')) or lamps.upper() in ('', 'NO VALUE'):
                return 'Off'

            return lamps
        except KeyError:
            return 'None'

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
        return self.observation_id()

    @astro_data_descriptor
    def is_ao(self):
        """
        Tells whether or not the data was taken with adaptive optics.

        Returns
        -------
        bool
            True if the data is AO, False otherwise.

        """
        return self.phu.get(self._keyword_for('ao_fold'), 'OUT') == 'IN'

    @astro_data_descriptor
    def is_coadds_summed(self):
        """
        Tells whether or not the co-adds have been summed.  If not, they
        have been averaged.

        At the Gemini level, this descriptor is hardcoded to True as it is
        the default at the observatory.

        Returns
        -------
        bool
            True if the data has been summed.  False if it has been averaged.

        """
        return True

    @astro_data_descriptor
    def local_time(self):
        """
        Returns the local time stored at the time of the observation.

        Returns
        -------
        datetime.datetime.time()
            Local time of the observation.

        """
        local_time = getattr(self.phu, self._keyword_for('local_time'))
        if re.match("^([012]\d)(:)([012345]\d)(:)(\d\d\.?\d*)$", local_time):
            return dateutil.parser.parse(local_time).time()
        else:
            raise ValueError("Invalid local_time: {!r}".format(local_time))

    @astro_data_descriptor
    def mdf_row_id(self):
        """
        Returns row ID from the MDF (Mask Definition File) table associated
        with the spectrum.  Applies to "cut" MOS or X-dispersed data.

        Returns
        -------
        int
            Row of the MDF associated with the extension.
        """
        tags = self.tags
        if 'IMAGE' in tags or 'PREPARED' not in tags:
            raise ValueError("This descriptor doesn't work on RAW or IMAGE files")

        return getattr(self.hdr, self._keyword_for('mdf_row_id'))

    @astro_data_descriptor
    def nominal_atmospheric_extinction(self):
        """
        Returns the nominal atmospheric extinction at observation airmass
        and bandpass.

        Returns
        -------
        float
            Nominal atmospheric extinction from model.

        """
        nom_ext_idx = (self.telescope(), self.filter_name(pretty=True))
        coeff = nominal_extinction.get(nom_ext_idx, 0.0)

        return coeff * (self.airmass() - 1.0)

    @astro_data_descriptor
    def qa_state(self):
        """
        Returns the Gemini quality assessment flags.

        Returns
        -------
        str
            Gemini quality assessment flags.

        """
        rawpireq = getattr(self.phu, self._keyword_for('raw_pi_requirements_met'))
        rawgemqa = getattr(self.phu, self._keyword_for('raw_gemini_qa'))
        pair = rawpireq.upper(), rawgemqa.upper()

        # Calculate the derived QA state
        ret_qa_state = "%s:%s" % (rawpireq, rawgemqa)
        if 'UNKNOWN' in pair:
                        ret_qa_state = "Undefined"
        elif pair == ('YES', 'USABLE'):
                        ret_qa_state = "Pass"
        elif pair == ('NO', 'USABLE'):
                        ret_qa_state = "Usable"
        elif rawgemqa.upper() == "BAD":
                        ret_qa_state = "Fail"
        elif 'CHECK' in pair:
                        ret_qa_state = "CHECK"
        else:
            ret_qa_state = "%s:%s" % (rawpireq, rawgemqa)

        return ret_qa_state

    def _raw_to_percentile(self, descriptor, raw_value):
        """
        Parses the Gemini constraint bands, and returns the percentile
        part as an integer.

        Parameters
        ----------
        descriptor : str
            The name of the descriptor calling this function.  For error
            reporting purposes.
        raw_value : str
            The sky constraint band.  (eg. 'IQ50')

        Returns
        -------
        int
            Percentile part of the Gemini constraint band.

        """
        val = parse_percentile(raw_value)
        if val is None:
            raise ValueError("Invalid value for {}: {!r}".format(descriptor, raw_value))
        return val

    @astro_data_descriptor
    def raw_bg(self):
        """
        Returns the BG, background brightness, of the observation.

        Returns
        -------
        str
            BG, background brightness, of the observation.

        """
        return self._raw_to_percentile('raw_bg', self.phu.RAWBG)

    @astro_data_descriptor
    def raw_cc(self):
        """
        Returns the CC, cloud coverage, of the observation.

        Returns
        -------
        str
            CC, cloud coverage of the observation.

        """
        return self._raw_to_percentile('raw_cc', self.phu.RAWCC)

    @astro_data_descriptor
    def raw_iq(self):
        """
        Returns the , image quality, of the observation.

        Returns
        -------
        str
            IQ, image quality, of the observation.

        """
        return self._raw_to_percentile('raw_iq', self.phu.RAWIQ)

    @astro_data_descriptor
    def raw_wv(self):
        """
        Returns the WV, water vapor, of the observation.

        Returns
        -------
        str
            WV, water vapor, of the observation.

        """
        return self._raw_to_percentile('raw_wv', self.phu.RAWWV)

    @astro_data_descriptor
    def requested_bg(self):
        """
        Returns the BG, background brightness, requested by the PI.

        Returns
        -------
        str
            BG, background brightness, requested by the PI.

        """
        return self._raw_to_percentile('requested_bg', self.phu.REQBG)

    @astro_data_descriptor
    def requested_cc(self):
        """
        Returns the CC, cloud coverage, requested by the PI.

        Returns
        -------
        str
            CC, cloud coverage, requested by the PI.

        """
        return self._raw_to_percentile('requested_cc', self.phu.REQCC)

    @astro_data_descriptor
    def requested_iq(self):
        """
        Returns the IQ, image quality, requested by the PI.

        Returns
        -------
        str
            IQ, image quality, requested by the PI.

        """
        return self._raw_to_percentile('requested_iq', self.phu.REQIQ)

    @astro_data_descriptor
    def requested_wv(self):
        """
        Returns the WV, water vapor, requested by the PI.

        Returns
        -------
        str
            WV, water vapor, requested by the PI.

        """
        return self._raw_to_percentile('requested_wv', self.phu.REQWV)

    @astro_data_descriptor
    def target_ra(self, offset=False, pm=True, icrs=False):
        """
        Returns the Right Ascension of the target in degrees. Optionally, the
        telescope offsets can be applied.  The proper motion can also be
        applied if requested.  Finally, the RA can be converted to ICRS
        coordinates.

        Parameters
        ----------
        offset : bool
            If True, applies the telescope offsets.
        pm : bool
            If True, applies proper motion parameters.
        icrs : bool
            If True, convert the RA to the ICRS coordinate system.

        Returns
        -------
        float
            Right Ascension of the target in degrees.

        """
        ra = getattr(self.phu, self._keyword_for('ra'))
        raoffset = self.phu.get('RAOFFSET', 0)
        targ_raoffset = self.phu.get('RATRGOFF', 0)
        pmra = self.phu.get('PMRA', 0)
        epoch = self.phu.get('EPOCH')
        frame = self.phu.get('FRAME')
        equinox = self.phu.get('EQUINOX')

        if offset:
            raoffset /= 3600.0
            targ_raoffset /= 3600.0
            raoffset += targ_raoffset
            raoffset /= math.cos(math.radians(self.target_dec(offset=True)))
            ra += raoffset

        if pm and pmra != 0:
            dt = self.ut_datetime()
            year = dt.year
            startyear = datetime.datetime(year, 1, 1, 0, 0, 0)
            # Handle leap year properly
            nextyear = datetime.datetime(year+1, 1, 1, 0, 0, 0)
            thisyear = nextyear - startyear
            sofar = dt - startyear
            fraction = sofar.total_seconds() / thisyear.total_seconds()
            obsepoch = year + fraction
            years = obsepoch - epoch
            pmra *= years
            pmra *= 15.0*math.cos(math.radians(self.target_dec(offset=True)))
            pmra /= 3600.0
            ra += pmra

        if icrs:
            ra, dec = toicrs(frame,
                    self.target_ra(offset=offset, pm=pm, icrs=False),
                    self.target_dec(offset=offset, pm=pm, icrs=False),
                    equinox=2000.0,
                    ut_datetime=dataset.ut_datetime()
                )

        return ra

    @astro_data_descriptor
    def target_dec(self, offset=False, pm=True, icrs=False):
        """
        Returns the Declination of the target in degrees. Optionally, the
        telescope offsets can be applied.  The proper motion can also be
        applied if requested.  Finally, the RA can be converted to ICRS
        coordinates.

        Parameters
        ----------
        offset : bool
            If True, applies the telescope offsets.
        pm : bool
            If True, applies proper motion parameters.
        icrs : bool
            If True, convert the Declination to the ICRS coordinate system.

        Returns
        -------
        float
            Declination of the target in degrees.

        """
        dec = getattr(self.phu, self._keyword_for('dec'))
        decoffset = self.phu.get('DECOFFSE', 0)
        targ_decoffset = self.phu.get('DECTRGOF', 0)
        pmdec = self.phu.get('PMDEC', 0)
        epoch = self.phu.get('EPOCH')
        frame = self.phu.get('FRAME')
        equinox = self.phu.get('EQUINOX')

        if offset:
            decoffset /= 3600.0
            targ_decoffset /= 3600.0
            dec += decoffset + targ_decoffset

        if pm and pmdec != 0:
            dt = self.ut_datetime()
            year = dt.year
            startyear = datetime.datetime(year, 1, 1, 0, 0, 0)
            # Handle leap year properly
            nextyear = datetime.datetime(year+1, 1, 1, 0, 0, 0)
            thisyear = nextyear - startyear
            sofar = dt - startyear
            fraction = sofar.total_seconds() / thisyear.total_seconds()
            obsepoch = year + fraction
            years = obsepoch - epoch
            pmdec *= years
            pmdec /= 3600.0
            dec += pmdec

        if icrs:
            ra, dec = toicrs(frame,
                    self.target_ra(offset=offset, pm=pm, icrs=False),
                    self.target_dec(offset=offset, pm=pm, icrs=False),
                    equinox=2000.0,
                    ut_datetime=dataset.ut_datetime()
                )

        return dec

    @astro_data_descriptor
    def ut_date(self):
        """
        Returns the UT date of the observation as a datetime object.

        Returns
        -------
        datetime.datetime
            UT date.

        """
        try:
            return self.ut_datetime(strict=True, dateonly=True)
        except AttributeError:
            raise LookupError("Can't find information to return a proper date")

    # TODO: Implement the ut_datetime function.
    @astro_data_descriptor
    def ut_datetime(self, strict=False, dateonly=False, timeonly=False):
        """
        Returns the UT date and time of the observation as a datetime object.

        Returns
        -------
        datetime.datetime
            UT date and time.

        """
        # Loop through possible header keywords to get the date (time may come
        # as a bonus with DATE-OBS)
        for kw in ['DATE-OBS', self._keyword_for('ut_date'), 'DATE', 'UTDATE']:
            utdate_hdr = self.phu.get(kw, '').strip()

            # Is this a full date+time string?
            if re.match("(\d\d\d\d-[01]\d-[0123]\d)(T)"
                        "([012]\d:[012345]\d:\d\d.*\d*)", utdate_hdr):
                return dateutil.parser.parse(utdate_hdr)

            # Did we just get a date?
            if re.match("\d\d\d\d-[01]\d-[0123]\d", utdate_hdr):
                break

            # Did we get a horrible early NIRI date: DD/MM/YY[Y]?
            match = re.match("([0123]\d)/([01]\d)/(\d\d+)", utdate_hdr)
            if match:
                y = 1900 + int(match.group(3))
                utdate_hdr = '{}-{}-{}'.format(y, match.group(2),
                                               match.group(1))
                break
            else:
                # Set any non-matching string to null
                utdate_hdr = ''

        # If we're here, utdate_hdr is either a date or empty
        # If we only need a date and we've got one, exit
        if dateonly and utdate_hdr:
            return dateutil.parser.parse('{} 00:00:00'.format(utdate_hdr)).date()

        # Now look for a time; again, several possible keywords
        for kw in [self._keyword_for('ut_time'), 'UT', 'TIME-OBS',
                   'STARTUT', 'UTSTART']:
            uttime_hdr = self.phu.get(kw, '').strip()
            if re.match("^([012]?\d)(:)([012345]?\d)(:)(\d\d?\.?\d*)$",
                        uttime_hdr):
                break
            else:
                uttime_hdr = ''

        # Now we've either got a time or a null string
        # If we only need a time and we've got one, exit
        if timeonly and uttime_hdr:
            return dateutil.parser.parse('2000-01-01 {}'.format(uttime_hdr)).time()

        # If we've got a date and a time, marry them and send them on honeymoon
        if utdate_hdr and uttime_hdr:
            return dateutil.parser.parse('{}T{}'.format(utdate_hdr,
                                                        uttime_hdr))

        # This is non-compliant data, maybe engineering or something
        # Try MJD_OBS
        mjd = self.phu.get('MJD_OBS', 0)
        if mjd > 1:
            mjdzero = datetime.datetime(1858, 11, 17, 0, 0, 0, 0, None)
            ut_datetime = mjdzero + datetime.timedelta(mjd)
            if dateonly:
                return ut_datetime.date()
            elif timeonly:
                return ut_datetime.time()
            else:
                return ut_datetime

        # Try OBSSTART
        obsstart = self.phu.get('OBSSTART')
        if obsstart:
            ut_datetime = dateutil.parser.parse(obsstart).replace(tzinfo=None)
        if dateonly:
            return ut_datetime.date()
        elif timeonly:
            return ut_datetime.time()
        else:
            return ut_datetime

        # Now we're getting desperate. Give up if strict=True
        if strict:
            return None

        # If we're missing a date, try to get it from the FRMNAME keyword or
        # the filename (.filename strips the path)
        if not utdate_hdr:
            values = self.hdr.get('FRMNAME', '') + [self.filename]
            for string in values:
                try:
                    year = string[1:5]
                    month = string[5:7]
                    day = string[7:9]
                    y = int(year)
                    m = int(month)
                    d = int(day)
                    if (y>1999 and m<13 and d<32):
                        utdate_hdr = '{}-{}-{}'.format(year, month, day)
                except (KeyError, ValueError, IndexError):
                    pass

        # If we're missing a time, set it to midnight
        if not uttime_hdr:
            uttime_hdr = '00:00:00'

        # Return something if we can fulfil the request
        if dateonly and utdate_hdr:
            return dateutil.parser.parse('{} 00:00:00'.format(utdate_hdr)).date()

        if timeonly and uttime_hdr:
            return dateutil.parser.parse('2000-01-01 {}'.format(uttime_hdr)).time()

        if utdate_hdr and uttime_hdr:
            return dateutil.parser.parse('{}T{}'.format(utdate_hdr, uttime_hdr))

        # Give up
        #raise LookupError("Can't find information to return requested date/time")
        return None

    @astro_data_descriptor
    def ut_time(self):
        """
        Returns the UT time of the observation as a datetime object.

        Returns
        -------
        datetime.datetime
            UT time.

        """
        try:
            return self.ut_datetime(strict=True, timeonly=True)
        except AttributeError:
            raise LookupError("Can't find information to return a proper time")

    @astro_data_descriptor
    def wavefront_sensor(self):
        """
        Returns the name of the wavefront sensor used for the observation.
        If more than one is being used, the names will be joined with '&'.

        Returns
        -------
        str
            Name of the wavefront sensor.

        """
        candidates = (
            ('AOWFS', self.phu.get("AOWFS_ST")),
            ('OIWFS', self.phu.get("OIWFS_ST")),
            ('PWFS1', self.phu.get("PWFS1_ST")),
            ('PWFS2', self.phu.get("PWFS2_ST")),
            ('GEMS', self.phu.get("GWFS1CFG"))
        )

        wavefront_sensors = [name for (name, value) in candidates if value == 'guiding']

        if not wavefront_sensors:
            raise ValueError("No probes are guiding")

        return '&'.join(sorted(wavefront_sensors))

    @astro_data_descriptor
    def wavelength_band(self):
        """
        Returns the name of the bandpass of the observation.  This is just
        to broadly know what type of data one is working with, eg. K band,
        H band, B band, etc.

        Returns
        -------
        str
            Name of the bandpass.

        """
        ctrl_wave = self.effective_wavelength(output_units='micrometers')

        def wavelength_diff((_, l)):
            return abs(l - ctrl_wave)
        band = min(wavelength_band.items(), key = wavelength_diff)[0]

        # TODO: This can't happen. We probably want to check against "None"
        if band is None:
            raise ValueError()

        return band

    @astro_data_descriptor
    def wavelength_reference_pixel(self):
        """
        Returns the wavelength reference pixel for each extension

        Returns
        -------
        list
            wavelength reference pixels
        """
        return getattr(self.hdr, self._keyword_for('wavelength_reference_pixel'))

    # TODO: Move everything below here to AstroDataFITS?
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
        return self._get_wcs_coords()[0]

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
        return self._get_wcs_coords()[1]

    @astro_data_descriptor
    def ra(self):
        """
        Returns the Right Ascension of the center of the field

        Returns
        -------
        float
            right ascension in degrees
        """
        return self.wcs_ra()

    @astro_data_descriptor
    def dec(self):
        """
        Returns the Declination of the center of the field.

        Returns
        -------
        float
            declination in degrees
        """
        return self.wcs_dec()

    def _get_wcs_coords(self, x=None, y=None):
        """
        Returns the RA and dec of the middle of the first extension
        or a specific pixel therein (added by CJS in case it's useful)

        Returns
        -------
        tuple
            (right ascension, declination)
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FITSFixedWarning)
            # header[0] is PHU, header[1] is first extension HDU
            # If no CTYPE1 in first HDU, try PHU
            if self.hdr.get('CTYPE1')[0] is None:
                wcs = WCS(self.header[0])
            else:
                wcs = WCS(self.header[1])
            if x is None or y is None:
                x, y = [0.5 * self.hdr.get(naxis)[0]
                        for naxis in ('NAXIS1','NAXIS2')]
            result = wcs.wcs_pix2world(x,y, 1)
        ra, dec = float(result[0]), float(result[1])

        # TODO: This isn't in old Gemini descriptors. Should it be?
        if 'NON_SIDEREAL' in self.tags:
            ra, dec = toicrs('APPT', ra, dec, ut_datetime=self.ut_datetime())

        return (ra, dec)

    # TODO: Move to AstroDataFITS? And deal with PCi_j/CDELTi keywords?
    def _get_wcs_pixel_scale(self):
        """
        Returns a list of pixel scales (in arcseconds), derived from the
        CD matrices of the image extensions

        Returns
        -------
        list
            List of pixel scales, one per extension
        """
        return [3600 * 0.5 * (math.sqrt(cd11 * cd11 + cd12 * cd12) +
                              math.sqrt(cd21 * cd21 + cd22 * cd22))
                for cd11, cd12, cd21, cd22 in zip(self.hdr.CD1_1,
                                                  self.hdr.CD1_2,
                                                  self.hdr.CD2_1,
                                                  self.hdr.CD2_2)]
