#
#                                                            Gemini Observatory
#
#                                                                        Dragons
#                                                             gemini_instruments
#                                                              gemini.adclass.py
# ------------------------------------------------------------------------------
import re
import math
import datetime
import dateutil.parser
from itertools import chain

import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord, Angle

from astrodata import AstroData
from astrodata import astro_data_tag
from astrodata import astro_data_descriptor
from astrodata import TagSet, Section

from .lookup import (wavelength_band, nominal_extinction,
                     filter_wavelengths, specphot_standards)

# NOTE: Temporary functions for test. gempy imports astrodata and
#       won't work with this implementation
from .. import gmu

# ------------------------------------------------------------------------------
# TODO: Some of these should go to AstroDataFITS
gemini_keyword_names = dict(
    airmass = 'AIRMASS',
    amp_read_area = 'AMPROA',
    ao_fold = 'AOFOLD',
    ao_seeing = 'AOSEEING',
    aperture_number = 'APERTURE',
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
    detector_x_bin = 'XBIN',
    detector_y_bin = 'YBIN',
    disperser = 'DISPERSR',
    dispersion = 'WDELTA',
    dispersion_axis = 'DISPAXIS',
    elevation = 'ELEVATIO',
    exposure_time = 'EXPTIME',
    filter_name = 'FILTER1',
    focal_plane_mask = 'FPMASK',
    gain = 'GAIN',
    gain_setting = 'GAINSET',
    gems = 'GWFS1CFG',
    grating = 'GRATING',
    grating_order = 'GRODER',
    group_id = 'GROUPID',
    local_time = 'LT',
    lyot_stop = 'LYOTSTOP',
    mdf_row_id = 'MDFROW',
    naxis1 = 'NAXIS1',
    naxis2 = 'NAXIS2',
    nominal_atmospheric_extinction = 'NOMATMOS',
    nominal_photometric_zeropoint = 'NOMPHOTZ',
    non_linear_level = 'NONLINEA',
    observation_epoch = 'OBSEPOCH',
    observation_id = 'OBSID',
    observation_mode = 'OBSMODE',
    oiwfs = 'OIWFS_ST',
    overscan_section = 'OVERSEC',
    pixel_scale = 'PIXSCALE',
    position_angle = 'PA',
    prism = 'PRISM',
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
    well_depth_setting = 'WELDEPTH',
    telescope_x_offset = 'XOFFSET',
    telescope_y_offset = 'YOFFSET',
)


def use_keyword_if_prepared(fn):
    """
    A decorator for descriptors. If decorated, the descriptor will bypass its
    main code on "PREPARED" data in favour of simply returning the value of
    the associated header keyword (as defined by the "_keyword_for" method)
    if this exists in all the headers (if the keyword is missing, it will
    execute the code in the descriptor method).
    """
    def gn(self):
        if "PREPARED" in self.tags:
            try:
                return self.hdr[self._keyword_for(fn.__name__)]
            except (KeyError, AttributeError):
                pass
        return fn(self)
    return gn


def get_specphot_name(ad):
    """
    Return the name of the specphotometric standard of which this AD object
    is an observation, or None if it is not an observation of a specphot.
    The name is returned in a whitespace-stripped, all-lowercase format
    corresponding to the filename containing the specphot data in
    geminidr.gemini.lookups.spectrophotometric_standards

    Parameters
    ----------
    ad: AstroData object which might be a specphot standard

    Returns
    -------
    str/None: name of the standard (or None if it's not a standard)
    """
    if ad.phu.get('OBSTYPE') != 'OBJECT':
        return
    target_name = ad.object().lower().replace(' ', '')
    try:
        target = SkyCoord(ad.target_ra(), ad.target_dec(), unit=u.deg)
    except (TypeError, ValueError):
        return
    try:
        dt = ad.ut_datetime() - datetime.datetime(2000, 1, 1, 12)
    except TypeError:
        dt = datetime.timedelta(days=3652.5)  # 10 years

    all_names, all_coords, all_pm_ra, all_pm_dec = [], [], [], []
    for name, (coords, pm_ra, pm_dec) in specphot_standards.items():
        all_names.append(name)
        all_coords.append(coords)
        all_pm_ra.append(pm_ra)
        all_pm_dec.append(pm_dec)
    c = SkyCoord(all_coords, unit=(u.hourangle, u.deg),
                 pm_ra_cosdec=all_pm_ra*u.mas/u.yr, pm_dec=all_pm_dec*u.mas/u.yr,
                 distance=10*u.pc)
    separations = target.separation(c.apply_space_motion(dt=dt)).arcsec
    i = separations.argmin()
    if separations[i] < 2 or separations[i] < 10 and all_names[i] == target_name:
        return all_names[i]


# ------------------------------------------------------------------------------
class AstroDataGemini(AstroData):
    __keyword_dict = gemini_keyword_names

    @staticmethod
    def _matches_data(source):
        obs = source[0].header.get('OBSERVAT', '').upper()
        tel = source[0].header.get('TELESCOP', '').upper()

        isGemini = (obs in ('GEMINI-NORTH', 'GEMINI-SOUTH')
                    or tel in ('GEMINI-NORTH', 'GEMINI-SOUTH'))
        return isGemini

    @astro_data_tag
    def _type_observatory(self):
        return TagSet(['GEMINI'])

    @astro_data_tag
    def _type_acquisition(self):
        if self.phu['OBSCLASS'] in ('acq', 'acqCal'):
            return TagSet(['ACQUISITION'])

    @astro_data_tag
    def _type_az(self):
        if self.phu.get('FRAME') == 'AZEL_TOPO':
            try:
                if self.airmass() == 1.0:
                    return TagSet(['AZEL_TARGET', 'AT_ZENITH'])
            except ValueError:
                pass
            return TagSet(['AZEL_TARGET'])

    @astro_data_tag
    def _type_fringe(self):
        if set (self.phu.keys()) & {'GIFRINGE', 'PROCFRNG'}:
            return TagSet(['FRINGE', 'CAL'])

    # GCALFLAT and the LAMPON/LAMPOFF are kept separated because the
    # PROCESSED status will cancel the tags for lamp status, but the
    # GCALFLAT is still needed
    @astro_data_tag
    def _type_gcalflat(self):
        gcallamp = self.phu.get('GCALLAMP')
        if gcallamp == 'IRhigh' or (gcallamp is not None and gcallamp.startswith('QH')):
            return TagSet(['GCALFLAT', 'FLAT', 'CAL'])

    @astro_data_tag
    def _type_gcal_lamp(self):
        if self.phu['INSTRUME'].startswith('GMOS'):
            return

        gcallamp = self.phu.get('GCALLAMP')
        if gcallamp in ('IRhigh', 'IRlow') or (gcallamp is not None and gcallamp.startswith('QH')):
            shut = self.phu.get('GCALSHUT')
            if shut == 'OPEN':
                return TagSet(['GCAL_IR_ON', 'LAMPON'], blocked_by=['PROCESSED'])
            elif shut == 'CLOSED':
                # GNIRS PINHOLE files have the QH lamp on, but this isn't really
                # a useful thing to indicate in tags. DB 20230906
                return TagSet(['GCAL_IR_OFF', 'LAMPOFF'], blocked_by=['PROCESSED',
                                                                      'PINHOLE'])
        elif self.phu.get('GCALLAMP') == 'No Value' and \
             self.phu.get('GCALSHUT') == 'CLOSED':
            return TagSet(['GCAL_IR_OFF', 'LAMPOFF'], blocked_by=['PROCESSED'])

    @astro_data_tag
    def _type_site(self):
        site = self.phu.get('OBSERVAT', '').upper()

        if site == 'GEMINI-NORTH':
            return TagSet(['NORTH'])
        elif site == 'GEMINI-SOUTH':
            return TagSet(['SOUTH'])

    @astro_data_tag
    def _type_mode(self):
        mode = self.phu.get(self._keyword_for('observation_mode'), '').upper()

        if mode:
            tags = [mode]
            if mode != 'IMAGE':
                # assume SPECT
                tags.append('SPECT')
            return TagSet(tags)

    @astro_data_tag
    def _type_nodandchop(self):
        if self.phu.get('DATATYPE') == "marked-nodandchop":
            return TagSet(['NODCHOP'])

    @astro_data_tag
    def _type_sidereal(self):
        frames = {self.phu.get('TRKFRAME'), self.phu.get('FRAME')}
        valid_frames = {'FK5', 'APPT'}

        # Check if the intersection of both sets is non-empty...
        if frames & valid_frames:
            try:
                dectrack, ratrack = float(self.phu['DECTRACK']), float(self.phu['RATRACK'])
                if dectrack == 0 and ratrack == 0:
                    return TagSet(['SIDEREAL'])
            except (ValueError, TypeError, KeyError):
                pass
            return TagSet(['NON_SIDEREAL'])

    @astro_data_tag
    def _type_bad_pixel_mask(self):
        if 'BPMASK' in self.phu or 'PROCBPM' in self.phu:
            return TagSet(['BPM', 'CAL', 'PROCESSED'],
                            blocks=['IMAGE', 'SPECT', 'NON_SIDEREAL',
                                    'AZEL_TARGET',
                                    'GCALFLAT', 'LAMPON', 'GCAL_IR_ON',
                                    'GCAL_IR_OFF', 'DARK'])

    @astro_data_tag
    def _type_mos_mask(self):
        if self.phu.get('OBSTYPE', '').upper() == "MASK":
            return TagSet(['MASK'], blocks=['IMAGE', 'SPECT'])

    @astro_data_tag
    def _status_raw(self):
        if 'GEM-TLM' not in self.phu:
            return TagSet(['RAW'])

    @astro_data_tag
    def _status_prepared(self):
        if any(('PREPAR' in kw) for kw in self.phu):
            return TagSet(['PREPARED'])
        else:
            return TagSet(['UNPREPARED'])

    @astro_data_tag
    def _status_overscan(self):
        found = []
        for pattern, tag in (('TRIMOVER', 'OVERSCAN_TRIMMED'), ('SUBOVER', 'OVERSCAN_SUBTRACTED')):
            if any((pattern in kw) for kw in self.phu):
                found.append(tag)
        if found:
            return TagSet(found)

    @astro_data_tag
    def _status_processed_cals(self):
        kwords = {'PROCARC', 'GBIAS', 'PROCBIAS', 'PROCDARK',
                  'GIFLAT', 'PROCFLAT', 'GIFRINGE', 'PROCFRNG',
                  'PROCPNHL', 'PROCSTND', 'PROCILLM'}

        if set(self.phu.keys()) & kwords:
            return TagSet(['PROCESSED'])

    @astro_data_tag
    def _status_processed_science(self):
        kwords = {'GMOSAIC', 'PROCSCI'}

        if self.phu['OBSTYPE'] == 'OBJECT' and set(self.phu.keys()) & kwords:
            return TagSet(['PROCESSED_SCIENCE', 'PROCESSED'], blocks=['RAW'])

    @astro_data_tag
    def _type_extracted(self):
        if 'EXTRACT' in self.phu:
            return TagSet(['EXTRACTED'])

    def _ra(self):
        """
        Parse RA from header.

        Utility method to pull the right ascension from the header, parsing text if appropriate.

        Returns
        -------
        float : right ascension in degrees, or None
        """
        ra = self.phu.get(self._keyword_for('ra'), None)
        if type(ra) == str:
            # maybe it's just a float
            try:
                return float(ra)
            except:
                try:
                    if not ra.endswith('hours') and not ra.endswith('degrees'):
                        rastr = f'{ra} hours'
                    else:
                        rastr = ra
                    return Angle(rastr).degree
                except:
                    self._logger.warning(f"Unable to parse RA from {ra}")
                    return None
        return ra

    def _dec(self):
        """
        Parse DEC from header.

        Utility method to pull the declination from the header, parsing text if appropriate.

        Returns
        -------
        float : declination in degrees, or None
        """
        dec = self.phu.get(self._keyword_for('dec'), None)
        if type(dec) == str:
            # maybe it's just a float
            try:
                return float(dec)
            except:
                try:
                    if not dec.endswith('degrees'):
                        decstr = f'{dec} degrees'
                    else:
                        decstr = dec
                    return Angle(decstr).degree
                except:
                    self._logger.warning(f"Unable to parse dec from {dec}")
                    return None
        return dec

    def _parse_section(self, keyword, pretty):
        try:
            value_filter = lambda x: (x.asIRAFsection() if pretty else x)
            process_fn = lambda x: (None if x is None else value_filter(x))
            # Dummy keyword FULLFRAME returns shape of full data array
            if keyword == 'FULLFRAME':
                def _encode_shape(shape):
                    if not shape:
                        return None
                    return Section(*chain.from_iterable([(0, npix) for npix in shape[::-1]]))
                if self.is_single:
                    sections = _encode_shape(self.shape)
                else:
                    sections = [_encode_shape(ext.shape) for ext in self]
            else:
                if self.is_single:
                    sections = Section.from_string(self.hdr.get(keyword))
                else:
                    sections = [Section.from_string(sec) if sec is not None else None for sec in self.hdr.get(keyword)]
            if self.is_single:
                return process_fn(sections)
            else:
                return [process_fn(raw) for raw in sections]
        except (KeyError, TypeError):
            return None

    def _may_remove_component(self, keyword, stripID, pretty):
        val = self.phu.get(keyword)
        if val and (stripID or pretty):
            return gmu.removeComponentID(val)
        return val

    @astro_data_descriptor
    def airmass(self):
        """
        Returns the airmass of the observation.

        Returns
        -------
        float
            Airmass value.
        """
        am = self.phu.get(self._keyword_for('airmass'), -1)
        if isinstance(am, str) and gmu.isBlank(am):
            return None
        return am if am >= 1 else None

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
            return self.phu[self._keyword_for('ao_seeing')]
        except KeyError:
            try:
                # If r_zero_val (Fried's parameter) is present,
                # a seeing estimate can be calculated (NOTE: Jo Thomas-Osip
                # is providing a reference for this calculation. Until then,
                # EJD checked using
                # http://www.ctio.noao.edu/~atokovin/tutorial/part1/turb.html )

                # Seeing at 0.5 micron
                rzv = self.phu[self._keyword_for('r_zero_val')]
                return (206265. * 0.98 * 0.5e-6) / (rzv * 0.01)
            except KeyError:
                return None

    @astro_data_descriptor
    def amp_read_area(self, pretty=False):
        """
        Returns the readout area of each amplifier, as a 0-based tuple or
        1-based string

        Returns
        -------
        list/(tuple or string)
            the amp readout areas
        """
        return self._parse_section(self._keyword_for('amp_read_area'), pretty)

    @astro_data_descriptor
    def array_name(self):
        """
        Returns the name of each array

        Returns
        -------
        list of str/str
            the array names
        """
        return self.hdr.get(self._keyword_for('array_name'))

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
        return self._parse_section(self._keyword_for('array_section'), pretty)

    @astro_data_descriptor
    def azimuth(self):
        """
        Returns the azimuth of the telescope, in degrees

        Returns
        -------
        float
            azimuth

        """
        return self.phu.get(self._keyword_for('azimuth'))

    @astro_data_descriptor
    def calibration_key(self):
        """
        Returns an object to be used as a key in the Calibrations dict.
        Multiple ADs can share a key but there can be only one of each type
        of calibration for each key. data_label() is the default.
        "_stack" is removed to avoid making a new request for a stacked
        frame, which will need the same calibration as the original.

        Returns
        -------
        string
            identifier

        """
        calk = self.data_label()
        try:
            return calk.replace('_stack', '')
        except AttributeError:
            return calk

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
        Returns the position angle of the Cassegrain rotator, in degrees.

        Returns
        -------
        float
            Position angle of the Cassegrain rotator.

        """
        crpa = self.phu.get(self._keyword_for('cass_rotator_pa'), 400)
        return crpa if abs(crpa) <= 360 else None

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

        # We assume that the central_wavelength keyword is in microns
        keyword = self._keyword_for('central_wavelength')
        wave_in_microns = self.phu.get(keyword, -1)
        # Convert to float if they saved it as a string
        wave_in_microns = float(wave_in_microns) if isinstance(wave_in_microns, str) else wave_in_microns
        if wave_in_microns < 0:
            return None
        return gmu.convert_units('micrometers', wave_in_microns,
                             output_units)

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
    def data_label(self):
        """
        Returns the data label of an observation.

        Returns
        -------
        str
            the observation's data label
        """
        return self.phu.get('DATALAB')

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
        return self._parse_section(self._keyword_for('data_section'), pretty)

    @astro_data_descriptor
    def dec(self):
        """
        Returns the Declination of the center of the field, in degrees.

        Returns
        -------
        float
            declination in degrees
        """
        dec = self.wcs_dec()
        if dec is None:
            dec = self._dec()
        return dec

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
    def detector_name(self, pretty=False):
        """
        Returns the name of the detector

        Returns
        -------
        str
            the detector name
        """
        return self.phu.get(self._keyword_for('detector_name'))

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
        return 'Fixed'

    @astro_data_descriptor
    def detector_rois_requested(self):
        """
        Returns the ROIs requested. Since most instruments don't have
        selectable ROIs, it returns None at the Gemini level

        Returns
        -------
        NoneType
            None
        """
        return None

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
        return self._parse_section(self._keyword_for('detector_section'), pretty)

    @astro_data_descriptor
    def detector_x_bin(self):
        """
        Returns the detector binning in the x-direction

        Returns
        -------
        int
            The detector binning
        """
        return self.phu.get(self._keyword_for('detector_x_bin'), 1)

    @astro_data_descriptor
    def detector_y_bin(self):
        """
        Returns the detector binning in the y-direction

        Returns
        -------
        int
            The detector binning
        """
        return self.phu.get(self._keyword_for('detector_y_bin'), 1)

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
        # Flipped if on bottom port
        return -offset if self.phu.get('INPORT')==1 else offset

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
            return self.phu.get('QOFFSET') / self.pixel_scale()
        except TypeError:  # either is None
            return None

    @astro_data_descriptor
    def disperser(self, stripID=False, pretty=False):
        """
        Returns the name of the disperser.  The "disperser" is a combination of
        all the dispersing elements along the light path.

        The component ID can be removed with either 'stripID' or 'pretty' set
        to True.

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
    def dispersion(self, asMicrometers=False, asNanometers=False, asAngstroms=False):
        """
        Returns the dispersion in meters per pixel as a list (one value per
        extension) or a float if used on a single-extension slice.  It is
        possible to control the units of wavelength using the input arguments.

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
        if self._keyword_for('dispersion') in self.hdr:
            dispersion = self.hdr[self._keyword_for('dispersion')]

        elif self._keyword_for('dispersion') in self.phu:
            dispersion = self.phu[self._keyword_for('dispersion')]

        else:
            dispersion = None

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

        if dispersion is not None:

            dispersion = gmu.convert_units('meters', dispersion, output_units)

            if not self.is_single:
                dispersion = [dispersion] * len(self)

        return dispersion

    @astro_data_descriptor
    def dispersion_axis(self):
        """
        Returns the axis along which the light is dispersed.

        Returns
        -------
        int
            Dispersion axis.
        """
        if 'PREPARED' not in self.tags:
            return None

        val = self.hdr.get(self._keyword_for('dispersion_axis'))

        def int_or_none(x):
            # cast to an int, but preserve None
            return None if x is None else int(x)

        try:
            # assume we have a list, but handle scalars below
            return [int_or_none(x) for x in val]
        except TypeError:
            # scalar
            return int_or_none(val)

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
        wave_in_microns = None
        tags = self.tags
        if 'IMAGE' in tags:
            filter_name = self.filter_name(pretty=True)
            for inst in ('*', self.instrument(generic=True)):
                try:
                    wave_in_microns = filter_wavelengths[inst, filter_name]
                except KeyError:
                    pass
        elif 'SPECT' in tags:
            wave_in_microns = self.central_wavelength(asMicrometers=True)
        return gmu.convert_units('micrometers', wave_in_microns, output_units)

    @astro_data_descriptor
    def elevation(self):
        """
        Returns the elevation of the telescope, in degrees

        Returns
        -------
        float
            elevation
        """
        return self.phu.get(self._keyword_for('elevation'))

    @astro_data_descriptor
    def exposure_time(self):
        """
        Returns the exposure time in seconds.

        Returns
        -------
        float
            Exposure time.
        """
        exposure_time = self.phu.get(self._keyword_for('exposure_time'), -1)
        if exposure_time < 0:
            return None

        if 'PREPARED' not in self.tags and self.is_coadds_summed():
            return exposure_time * self.coadds()
        else:
            return exposure_time

    @astro_data_descriptor
    def filter_name(self, stripID=False, pretty=False, keepID=False):
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
        f1 = self._may_remove_component('FILTER1', stripID, pretty)
        f2 = self._may_remove_component('FILTER2', stripID, pretty)
        if f1 is None or f2 is None:
            return None

        if pretty:
            filter_comps = []
            for fn in (f1, f2):
                # Not interested in clear or neutral density filters
                if not ("open" in fn.lower() or "Clear" in fn or
                            fn.lower().startswith('nd')):
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
        list of floats/float
            Gains used for the observation
        """
        return self.hdr.get(self._keyword_for('gain'))

    @astro_data_descriptor
    def gain_setting(self):
        """
        Returns the gain setting for this observation (e.g., 'high', 'low')

        Returns
        -------
        str
            the gain setting
        """
        return self.phu.get(self._keyword_for('gain_setting'))

    @astro_data_descriptor
    def gcal_lamp(self):
        """
        Returns the name of the GCAL lamp being used, or "Off" if no lamp is
        in used.  This applies to flats and arc observations when a lamp is
        used.  For other types observation, None is returned.

        Returns
        -------
        str
            Name of the GCAL lamp being used, or "Off" if not in use.
        """
        try:
            lamps, shut = self.phu['GCALLAMP'], self.phu['GCALSHUT']
            if (shut.upper() == 'CLOSED' and lamps.upper() in
                ('IRHIGH', 'IRLOW')) or lamps.upper() in ('', 'NO VALUE'):
                return 'Off'
            return lamps
        except KeyError:
            return None

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
    def instrument(self, generic=False):
        """
        Returns the name of the instrument making the observation

        Parameters
        ----------
        generic: boolean
            If set, don't specify the specific instrument if there are clones
            (but that is handled by the instrument-level descriptors)

        Returns
        -------
        str
            instrument name
        """
        return self.phu.get(self._keyword_for('instrument'))

    @astro_data_descriptor
    def is_ao(self):
        """
        Tells whether or not the data was taken with adaptive optics.

        Returns
        -------
        bool
            True if the data is AO, False otherwise.
        """
        # If the keyword's not there, assume the mirror is out
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
    def is_in_adu(self):
        """
        Tells whether the data are in ADU (likely to be superseded by use of
        NDData's unit attribute)

        Returns
        -------
        bool
            True if the data are in ADU
        """
        units = self.hdr.get('BUNIT', 'ADU')
        if self.is_single:
            return units.upper() == 'ADU'
        units = {u.upper() for u in units}
        if len(units) > 1:
            raise ValueError("Not all extensions appear to have the same units")
        return units.pop() == 'ADU'

    @astro_data_descriptor
    def local_time(self):
        """
        Returns the local time stored at the time of the observation.

        Returns
        -------
        datetime.datetime.time()
            Local time of the observation.
        """
        try:
            local_time = self.phu[self._keyword_for('local_time')]
            return dateutil.parser.parse(local_time).time()
        except (ValueError, TypeError, KeyError):
            return None

    @astro_data_descriptor
    def lyot_stop(self, stripID=False, pretty=False):
        """
        Returns the Lyot stop used for the observation

        Parameters
        ----------
        stripID : bool
            If True, removes the component ID.
        pretty : bool
            Same as for stripID.

        Returns
        -------
        str
            Lyot stop used for the observation
        """
        return self._may_remove_component(self._keyword_for('lyot_stop'),
                                          stripID, pretty)

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
            return None
        return self.hdr.get(self._keyword_for('mdf_row_id'))

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
        am = self.airmass()
        return coeff * (self.airmass() - 1.0) if am else None

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
        return self.hdr.get(self._keyword_for('nominal_photometric_zeropoint'))

    @astro_data_descriptor
    def non_linear_level(self):
        """
        Returns the level at which the data become non-linear, in ADU.
        This is expected to be overridden by the individual instruments,
        so at the Gemini level it returns the values of the NONLINEA keywords
        (or None)

        Returns
        -------
        int/list
            non-linearity level level in ADU
        """
        return self.hdr.get(self._keyword_for('non_linear_level'))

    @astro_data_descriptor
    def observation_class(self):
        """
        Returns the class of an observation, e.g., 'science', 'acq', 'dayCal'.

        Returns
        -------
        str
            the observation class
        """
        return self.phu.get('OBSCLASS')

    @astro_data_descriptor
    def observation_id(self):
        """
        Returns the ID of an observation.

        Returns
        -------
        str
            the observation ID
        """
        return self.phu.get('OBSID')

    @astro_data_descriptor
    def observation_epoch(self):
        """
        Returns the observation's epoch.

        Returns
        -------
        str
            the observation's epoch
        """
        return self.phu.get(self._keyword_for('observation_epoch'))

    @astro_data_descriptor
    def observation_type(self):
        """
        Returns the type of an observation, e.g., 'OBJECT', 'FLAT', 'ARC'.

        Returns
        -------
        str
            the observation type
        """
        return self.phu.get('OBSTYPE')

    @astro_data_descriptor
    def overscan_section(self, pretty=False):
        """
        Returns the section covered by the overscan regions relative to the
        detector frame. If pretty is False, a tuple of 0-based coordinates
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

        str/list of str
            Position of extension(s) using an IRAF section format (1-based)
        """
        return self._parse_section(self._keyword_for('overscan_section'), pretty)

    @astro_data_descriptor
    def pixel_scale(self):
        """
        Returns the image scale in arcseconds per pixel, as an average over
        the extensions

        Returns
        -------
        float
            the pixel scale
        """
        return self._get_wcs_pixel_scale(mean=True)

    @astro_data_descriptor
    def position_angle(self):
        """
        Returns the position angle of the instruement

        Returns
        -------
        float
            the position angle (East of North) of the +ve y-direction
        """
        return self.phu[self._keyword_for('position_angle')]

    @astro_data_descriptor
    def program_id(self):
        """
        Returns the ID of the program the observation was taken for

        Returns
        -------
        str
            the program ID
        """
        return self.phu.get('GEMPRGID')

    @astro_data_descriptor
    def pupil_mask(self, stripID=False, pretty=False):
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
        return self._may_remove_component(self._keyword_for('pupil_mask'),
                                   stripID, pretty)

    @astro_data_descriptor
    def qa_state(self):
        """
        Returns the Gemini quality assessment flags.

        Returns
        -------
        str
            Gemini quality assessment flags.
        """
        rawpireq = self.phu.get(self._keyword_for('raw_pi_requirements_met'),
                                'UNKNOWN')
        rawgemqa = self.phu.get(self._keyword_for('raw_gemini_qa'), 'UNKNOWN')
        pair = rawpireq.upper(), rawgemqa.upper()

        # Calculate the derived QA state
        ret_qa_state = "{}:{}".format(rawpireq, rawgemqa)
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
        return ret_qa_state

    @astro_data_descriptor
    def ra(self):
        """
        Returns the Right Ascension of the center of the field, in degrees.

        Returns
        -------
        float
            right ascension in degrees
        """
        ra = self.wcs_ra()
        if ra is None:
            ra = self._ra()
        return ra

    @astro_data_descriptor
    def raw_bg(self):
        """
        Returns the BG percentile band of the observation.  BG refers to the
        sky/background brightness.

        Returns
        -------
        str
            BG percentile band of the observation.
        """
        return self._raw_to_percentile('raw_bg', self.phu.get('RAWBG'))

    @astro_data_descriptor
    def raw_cc(self):
        """
        Returns the CC percentile band of the observation.  CC refers to the
        cloud coverage.

        Returns
        -------
        str
            CC percentile band of the observation.
        """
        return self._raw_to_percentile('raw_cc', self.phu.get('RAWCC'))

    @astro_data_descriptor
    def raw_iq(self):
        """
        Returns the IQ percentile band of the observation.  IQ refers to the
        image quality or seeing.

        Returns
        -------
        str
            IQ percentile band of the observation.
        """
        return self._raw_to_percentile('raw_iq', self.phu.get('RAWIQ'))

    @astro_data_descriptor
    def raw_wv(self):
        """
        Returns the WV percentile band of the observation.  WV refers to the
        water vapor.

        Returns
        -------
        str
            WV percentile band of the observation.
        """
        return self._raw_to_percentile('raw_wv', self.phu.get('RAWWV'))

    @astro_data_descriptor
    def read_mode(self):
        """
        Returns the readout mode used for the observation

        Returns
        -------
        str
            the read mode used
        """
        return self.phu.get(self._keyword_for('read_mode'))

    @astro_data_descriptor
    def read_noise(self):
        """
        Returns the read noise in electrons for each extension. A list is
        returned unless called on a single-extension slice, when a float

        Returns
        -------
        float/list of floats
            the read noise
        """
        return self.hdr.get(self._keyword_for('read_noise'))

    @astro_data_descriptor
    def read_speed_setting(self):
        """
        Returns the read speed setting for the observation

        Returns
        -------
        str
            the read speed setting
        """
        return self.phu.get(self._keyword_for('read_speed_setting'))

    @astro_data_descriptor
    def requested_bg(self):
        """
        Returns the BG percentile band requested by the PI.  BG refers to the
        sky/background brightness.

        Returns
        -------
        str
            BG percentile band requested by the PI.
        """
        return self._raw_to_percentile('requested_bg', self.phu.get('REQBG'))

    @astro_data_descriptor
    def requested_cc(self):
        """
        Returns the CC percentile band requested by the PI.  CC refers to the
        cloud coverage.

        Returns
        -------
        str
            CC percentile band requested by the PI.
        """
        return self._raw_to_percentile('requested_cc', self.phu.get('REQCC'))

    @astro_data_descriptor
    def requested_iq(self):
        """
        Returns the IQ percentile band requested by the PI.  IQ refers to the
        image quality or seeing.

        Returns
        -------
        str
            IQ percentile band requested by the PI.
        """
        return self._raw_to_percentile('requested_iq', self.phu.get('REQIQ'))

    @astro_data_descriptor
    def requested_wv(self):
        """
        Returns the WV percentile band requested by the PI.  WV refers to the
        water vapor.

        Returns
        -------
        str
            WV percentile band requested by the PI.
        """
        return self._raw_to_percentile('requested_wv', self.phu.get('REQWV'))

    @astro_data_descriptor
    def saturation_level(self):
        """
        Returns the saturation level of the data, in the units of the data.
        This is expected to be overridden by the individual instruments,
        so at the Gemini level it returns the values of the SATLEVEL keyword
        (or None).

        Returns
        -------
        list/float
            saturation level (in units of the data)
        """
        return self.hdr.get(self._keyword_for('saturation_level'))

    @astro_data_descriptor
    def slit(self):
        """
        Returns the name of the entrance slit used for the observation

        Returns
        -------
        str
            the slit name
        """
        return self.phu.get(self._keyword_for('slit'))

    @astro_data_descriptor
    def slit_width(self):
        """
        Returns the width of the slit in arcseconds

        Returns
        -------
        float
            the slit width in arcseconds
        """
        if 'SPECT' in self.tags:
            raise NotImplementedError("A slit width is not defined for this instrument")
        return None

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

        try:
            ra = self._ra()
        except KeyError:
            return None

        if isinstance(ra, str) and gmu.isBlank(ra):
            return None

        raoffset = self.phu.get('RAOFFSET', 0)
        targ_raoffset = self.phu.get('RATRGOFF', 0)
        pmra = self.phu.get('PMRA', 0)
        epoch = self.phu.get('EPOCH', 2000.0)
        frame = self.phu.get('FRAME', 'FK5')

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
            # PMRA is to be in time-seconds/yr so cos(dec) term is not needed
            pmra *= 15.0 #*math.cos(math.radians(self.target_dec(offset=True)))
            pmra /= 3600.0
            ra += pmra

        if icrs:
            ra, dec = gmu.toicrs(frame,
                    self.target_ra(offset=offset, pm=pm, icrs=False),
                    self.target_dec(offset=offset, pm=pm, icrs=False),
                    equinox=2000.0,
                    ut_datetime=self.ut_datetime()
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
        try:
            dec = self._dec()
        except KeyError:
            return None

        if isinstance(dec, str) and gmu.isBlank(dec):
            return None

        decoffset = self.phu.get('DECOFFSE', 0)
        targ_decoffset = self.phu.get('DECTRGOF', 0)
        pmdec = self.phu.get('PMDEC', 0)
        epoch = self.phu.get('EPOCH', 2000.0)
        frame = self.phu.get('FRAME', 'FK5')

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
            ra, dec = gmu.toicrs(frame,
                    self.target_ra(offset=offset, pm=pm, icrs=False),
                    self.target_dec(offset=offset, pm=pm, icrs=False),
                    equinox=2000.0,
                    ut_datetime = self.ut_datetime()
                )

        return dec

    @astro_data_descriptor
    def telescope_x_offset(self):
        """
        Returns the telescope offset along the telescope x-axis, in arcseconds.

        Returns
        -------
        float
            the telescope offset along the telescope x-axis (arcseconds)
        """
        return self.phu.get(self._keyword_for('telescope_x_offset'))

    @astro_data_descriptor
    def telescope_y_offset(self):
        """
        Returns the telescope offset along the telescope y-axis, in arcseconds.

        Returns
        -------
        float
            the telescope offset along the telescope y-axis (arcseconds)
        """
        return self.phu.get(self._keyword_for('telescope_y_offset'))

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
            return None

    @astro_data_descriptor
    def ut_datetime(self, strict=False, dateonly=False, timeonly=False):
        """
        Returns the UT date and/or time of the observation as a datetime
        or date or time object.

        Parameters
        ----------
        strict: bool
            if True, only information in the header can be used
        dateonly: bool
            if True, return a date object with just the date
        timeonly: bool
            if True, return a time object with just the time

        Returns
        -------
        datetime.datetime / datetime.date / datetime.time
            UT date and/or time
        """
        # Loop through possible header keywords to get the date (time may come
        # as a bonus with DATE-OBS)
        for kw in ['DATE-OBS', self._keyword_for('ut_date'), 'DATE', 'UTDATE']:
            utdate_hdr = self.phu.get(kw, '').strip()

            # Is this a full date+time string?
            if re.match(r"(\d\d\d\d-[01]\d-[0123]\d)(T)"
                        r"([012]\d:[012345]\d:\d\d.*\d*)", utdate_hdr):
                ut_datetime = dateutil.parser.parse(utdate_hdr)
                if dateonly:
                    return ut_datetime.date()
                elif timeonly:
                    return ut_datetime.time()
                return ut_datetime

            # Did we just get a date?
            if re.match(r"\d\d\d\d-[01]\d-[0123]\d", utdate_hdr):
                break

            # Did we get a horrible early NIRI date: DD/MM/YY[Y]?
            match = re.match(r"([0123]\d)/([01]\d)/(\d\d+)", utdate_hdr)
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
            if re.match(r"^([012]?\d)(:)([012345]?\d)(:)(\d\d?\.?\d*)$",
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
        mjd = self.phu.get('MJD-OBS', 0)
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
            ut_datetime = dateutil.parser.parse(obsstart.strip()).replace(tzinfo=None)
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
            return None

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
        )

        wavefront_sensors = [name for (name, value) in candidates
                             if value == 'guiding']
        if self.phu.get('GWFS1CFG') is not None:
            wavefront_sensors.append('GEMS')

        return '&'.join(sorted(wavefront_sensors)) if wavefront_sensors else None

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
        if ctrl_wave is None:
            return None

        def wavelength_diff(pair):
            _, l = pair
            return abs(l - ctrl_wave)
        band = min(wavelength_band.items(), key=wavelength_diff)[0]
        return band

    # TODO: Move RA/dec stuff to AstroDataFITS?
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
        # Return None if the WCS isn't sky coordinates
        try:
            return self._get_wcs_coords()['lon']
        except (KeyError, TypeError):
            return None

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
        # Return None if the WCS isn't sky coordinates
        try:
            return self._get_wcs_coords()['lat']
        except (KeyError, TypeError):
            return None

    @astro_data_descriptor
    def well_depth_setting(self):
        """
        Returns a string describing the well-depth setting of the instrument

        Returns
        -------
        str
            the well-depth setting
        """
        return self.phu.get(self._keyword_for('well_depth_setting'))

    def _get_wcs_coords(self):
        """
        Returns the RA and dec of the location around which the celestial
        sphere is being projected (CRVALi in a FITS representation)

        Returns
        -------
        dict
            {'lon': right ascension, 'lat': declination} plus other coords
        """
        wcs = self.wcs if self.is_single else self[0].wcs
        if wcs is None:
            return None

        ra = dec = None
        coords = {name: None for name in wcs.output_frame.axes_names}
        for m in wcs.forward_transform:
            try:
                ra = m.lon.value
                dec = m.lat.value
            except AttributeError:
                pass

        if 'NON_SIDEREAL' in self.tags and ra is not None and dec is not None:
            ra, dec = gmu.toicrs('APPT', ra, dec, ut_datetime=self.ut_datetime())

        coords["lon"] = ra
        coords["lat"] = dec
        return coords

    # TODO: Move to AstroDataFITS?
    def _get_wcs_pixel_scale(self, mean=True):
        """
        Returns a list of pixel scales (in arcseconds), derived from the
        CD matrices of the image extensions

        Parameters
        ----------
        mean: bool
            if set, return a single value across all extensions

        Returns
        -------
        list of floats/float
            List of pixel scales, one per extension
        """
        def empirical_pixel_scale(ext):
            """Brute-force calculation of pixel scale"""
            if ext.wcs is None:
                return None
            yc, xc = [0.5 * l for l in ext.shape]
            ra, dec = ext.wcs([xc, xc, xc+1], [yc, yc+1, yc])[-2:]
            cosdec = math.cos(dec[0] * np.pi / 180)
            a = (ra[2] - ra[0]) * cosdec
            b = (ra[1] - ra[0]) * cosdec
            c = dec[2] - dec[0]
            d = dec[1] - dec[0]
            return 3600 * np.sqrt(abs(a * d - b * c))

        if self.is_single:
            try:
                return 3600 * np.sqrt(abs(np.linalg.det(self.wcs.forward_transform['cd_matrix'].matrix)))
            except (IndexError, AttributeError):
                return empirical_pixel_scale(self)

        pixel_scale_list = []
        for ext in self:
            if len(ext.shape) < 2:
                return None
            try:
                pixel_scale_list.append(3600 * np.sqrt(abs(np.linalg.det(ext.wcs.forward_transform['cd_matrix'].matrix))))
            except (IndexError, AttributeError) as iae:
                scale = empirical_pixel_scale(ext)
                if scale is not None:
                    pixel_scale_list.append(scale)
        if mean:
            if pixel_scale_list:
                return np.mean(pixel_scale_list)
            return None
        return pixel_scale_list

    def _grating(self):
        """
        Returns the grating used for the observation

        Returns
        -------
        str
            Grating used for the observation
        """
        return self.phu.get(self._keyword_for('grating'))

    def _grating_order(self):
        """
        Returns the grating order for the observation

        Returns
        -------
        int
            Grating order for the observation if it is spectra, else None
        """
        if 'SPECT' in self.tags:
            try:
                grating_order = self.phu.get(self._keyword_for('grating_order'))
                if grating_order is not None:
                    return int(grating_order)
            except:
                pass
        # If it isn't SPECT, or has no grating order, this is not relevant and we return None
        return None

    def _prism(self):
        """
        Returns the name of the prism used for the observation

        Returns
        -------
        str
            the prism
        """
        return self.phu.get(self._keyword_for('prism'))


    def _raw_to_percentile(self, descriptor, raw_value):
        """
        Parses the Gemini constraint bands, and returns the percentile
        part as an integer.

        Parameters
        ----------
        descriptor : str
            The name of the descriptor calling this function.  For error
            reporting purposes. [Not used since we want to return None]
        raw_value : str
            The sky constraint band.  (eg. 'IQ50')

        Returns
        -------
        int
            Percentile part of the Gemini constraint band.
        """
        val = gmu.parse_percentile(raw_value)
        return val
