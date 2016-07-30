import re
import datetime
import dateutil.parser

import pywcs

from astrodata import AstroDataFits, astro_data_tag, TagSet
from astrodata import factory, simple_descriptor_mapping, keyword
from .lookup import wavelength_band, nominal_extinction, filter_wavelengths

# NOTE: Temporary functions for test. gempy imports astrodata and
#       won't work with this implementation
from ..gmu import *

gemini_direct_keywords = dict(
    ao_fold = keyword("AOFOLD"),
    array_name = keyword("ARRAYNAM"),
    azimuth = keyword("AZIMUTH"),
    bias_image = keyword("BIASIM"),
    bunit = keyword("BUNIT"),
    cd11 = keyword("CD1_1"),
    cd12 = keyword("CD1_2"),
    cd21 = keyword("CD2_1"),
    cd22 = keyword("CD2_2"),
    dark_image = keyword("DARKIM"),
    data_label = keyword("DATALAB"),
#    dec = keyword("DEC"),
    detector_name = keyword("DETNAME"),
    detector_roi_setting = keyword("DROISET", default="Fixed"),
    detector_rois_requested = keyword("DROIREQ"),
    detector_x_bin = keyword("XCCDBIN", default=1),
    detector_y_bin = keyword("YCCDBIN", default=1),
    dispersion = keyword("WDELTA"),
    elevation = keyword("ELEVATIO"),
    gain = keyword("GAIN"),
    gain_setting = keyword("GAINSET"),
    grating = keyword("GRATING"),
    lyot_stop = keyword("LYOTSTOP"),
    naxis1 = keyword("NAXIS1"),
    naxis2 = keyword("NAXIS2"),
    nod_count = keyword("NODCOUNT"),
    nod_pixels = keyword("NODPIX"),
    nominal_photometric_zeropoint = keyword("NOMPHOTZ"),
    non_linear_level = keyword("NONLINEA"),
    observation_class = keyword("OBSCLASS"),
    observation_epoch = keyword("OBSEPOCH", coerce_with=str),
    observation_id = keyword("OBSID"),
    observation_type = keyword("OBSTYPE"),
    overscan_section = keyword("OVERSSEC"),
    pixel_scale = keyword("PIXSCALE"),
    prism = keyword("PRISM"),
    program_id = keyword("GEMPRGID"),
    pupil_mask = keyword("PUPILMSK"),
    r_zero_val = keyword("RZEROVAL"),
#    ra = keyword("RA"),
    raw_central_wavelength = keyword("CWAVE"),
    raw_gemini_qa = keyword("RAWGEMQA"),
    raw_pi_requirements_met = keyword("RAWPIREQ"),
    read_mode = keyword("READMODE"),
    read_noise = keyword("RDNOISE"),
    read_speed_setting = keyword("RDSPDSET"),
    saturation_level = keyword("SATLEVEL"),
    slit = keyword("SLIT"),
    wavelength = keyword("WAVELENG"),
    wavelength_reference_pixel = keyword("WREFPIX", on_ext=True),
    well_depth_setting = keyword("WELDEPTH"),
    x_offset = keyword("XOFFSET"),
    y_offset = keyword("YOFFSET"),
)

@simple_descriptor_mapping(**gemini_direct_keywords)
class AstroDataGemini(AstroDataFits):
    @staticmethod
    def _matches_data(data_provider):
        obs = data_provider.header[0].get('OBSERVAT').upper()
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
            return TagSet(['CAL', 'FRINGE'])

    # GCALFLAT and the LAMPON/LAMPOFF are kept separated because the
    # PROCESSED status will cancel the tags for lamp status, but the
    # GCALFLAT is still needed
    @astro_data_tag
    def _type_gcalflat(self):
        if self.phu.GCALLAMP == 'IRhigh':
            return TagSet(['GCALFLAT'])

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

    def _some_section(self, descriptor_name, keyword, pretty):
        try:
            value_filter = (str if pretty else sectionStrToIntList)
            ret = [(None if raw is None else value_filter(raw))
                    for raw in getattr(self.hdr, keyword)]
            if len(ret) == 1:
                return ret[0]
            return ret
        except KeyError:
            raise AttributeError("No {} information".format(descriptor_name))

    def _may_remove_component(self, keyword, stripID, pretty):
        val = getattr(self.phu, keyword)
        if stripID or pretty:
            return removeComponentID(val)
        return val

    @property
    def instrument_name(self):
        return self.instrument().upper()

    def airmass(self):
        am = self.phu.AIRMASS

        if am < 1:
            raise ValueError("Can't have less than 1 airmass!")

        return float(am)

    def ao_seeing(self):
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
                rzv = self.r_zero_val()
                return (206265. * 0.98 * 0.5e-6) / (rzv * 0.01)
            except KeyError:
                raise AttributeError("There is no information about AO seeing")

    def array_section(self, pretty=False):
        return self._some_section('array_section', 'ARRAYSEC', pretty)

    def camera(self, stripID=False, pretty=False):
        return self._may_remove_component('CAMERA', stripID, pretty)

    def cass_rotator_pa(self):
        val = float(self.phu.CRPA)
        if val < -360 or val > 360:
            raise ValueError("Invalid CRPA value: {}".format(val))
        return val

    # TODO: Allow for unit conversion
    def central_wavelength(self):
        val = self.raw_central_wavelength()
        if val < 0:
            raise ValueError("Invalid CWAVE value: {}".format(val))

        # We assume that raw_central_wavelength returns micrometers.
        return val / 1e-6

    def coadds(self):
        return int(self.phu.get('COADDS', 1))

    def data_section(self, pretty=False):
        return self._some_section('data_section', 'DATASEC', pretty)

    def decker(self, stripID=False, pretty=False):
        return self._may_remove_component('DECKER', stripID, pretty)

    def detector_section(self, pretty=False):
        return self._some_section('detector_section', 'DETSEC', pretty)

    def disperser(self, stripID=False, pretty=False):
        return self._may_remove_component('DISPERSR', stripID, pretty)

    def dispersion_axis(self):
        # Keyword: DISPAXIS
        tags = self.tags
        if 'IMAGE' in tags or 'PREPARED' not in tags:
            raise ValueError("This descriptor doesn't work on RAW or IMAGE files")

        # TODO: We may need to sort out Nones here...
        return [int(dispaxis) for dispaxis in self.hdr.DISPAXIS]

    def effective_wavelength(self):
        # TODO: We need to return the appropriate output units
        tags = self.tags
        if 'IMAGE' in tags:
            inst = self.instrument()
            filter_name = self.filter_name(pretty=True)
            for inst in (self.instrument(), '*'):
                try:
                    return filter_wavelengths[inst, filter_name]
                except KeyError:
                    pass
            raise KeyError("Can't find the wavelenght for this filter in the look-up table")
        elif 'SPECT' in tags:
            return self.central_wavelength()

    def exposure_time(self):
        exposure_time = self.phu.EXPTIME
        if exposure_time < 0:
            raise ValueError("Invalid exposure time: {}".format(exposure_time))

        if 'PREPARED' in self.tags and self.is_coadds_summed():
            return exposure_time * self.coadds()
        else:
            return exposure_time

    def filter_name(self, stripID=False, pretty=False):
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

    def focal_plane_mask(self, stripID=False, pretty=False):
        self._may_remove_component('FPMASK', stripID, pretty)

    def gcal_lamp(self):
        try:
            lamps, shut = self.phu.GCALLAMP, self.phu.GCALSHUT
            if (shut.upper() == 'CLOSED' and lamps.upper() in ('IRHIGH', 'IRLOW')) or lamps.upper() in ('', 'NO VALUE'):
                return 'Off'

            return lamps
        except KeyError:
            return 'None'

    def group_id(self):
        return self.observation_id()

    def is_ao(self):
        try:
            return self.ao_fold() == 'IN'
        except KeyError:
            return False

    def is_coadds_summed(self):
        return True

    def local_time(self):
        local_time = self.phu.LT
        if re.match("^([012]\d)(:)([012345]\d)(:)(\d\d\.?\d*)$", local_time):
            return dateutil.parser.parse(local_time).time()
        else:
            raise ValueError("Invalid local_time: {!r}".format(local_time))

    def mdf_row_id(self):
        # Keyword: MDFROW
        raise NotImplementedError("mdf_row_id needs types/tags...")

    def nominal_atmospheric_extinction(self):
        nom_ext_idx = (self.telescope(), self.filter_name(pretty=True))
        coeff = nominal_extinction.get(nom_ext_idx, 0.0)

        return coeff * (self.airmass() - 1.0)

    def qa_state(self):
        rawpireq = self.raw_pi_requirements_met()
        rawgemqa = self.raw_gemini_qa()
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
        val = parse_percentile(raw_value)
        if val is None:
            raise ValueError("Invalid value for {}: {!r}".format(descriptor, raw_value))
        return val

    def raw_bg(self):
        return self._raw_to_percentile('raw_bg', self.phu.RAWBG)

    def raw_cc(self):
        return self._raw_to_percentile('raw_cc', self.phu.RAWCC)

    def raw_iq(self):
        return self._raw_to_percentile('raw_iq', self.phu.RAWIQ)

    def raw_wv(self):
        return self._raw_to_percentile('raw_wv', self.phu.RAWWV)

    def requested_bg(self):
        return self._raw_to_percentile('raw_bg', self.phu.REQBG)

    def requested_cc(self):
        return self._raw_to_percentile('raw_cc', self.phu.REQCC)

    def requested_iq(self):
        return self._raw_to_percentile('raw_iq', self.phu.REQIQ)

    def requested_wv(self):
        return self._raw_to_percentile('raw_wv', self.phu.REQWV)

    def target_ra(self, offset=False, pm=True, icrs=False):
        ra = self.ra()
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

    def target_dec(self, offset=False, pm=True, icrs=False):
        dec = self.dec()
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

    def ut_date(self):
        try:
            return self.ut_datetime(strict=True, dateonly=True).date()
        except AttributeError:
            raise LookupError("Can't find information to return a proper date")

    def ut_datetime(self, strict=False, dateonly=False, timeonly=False):
        raise NotImplementedError("Getting ut_datetime is stupidly complicated. Will be implemented later")

    def ut_time(self):
        try:
            return self.ut_datetime(strict=True, timeonly=True).time()
        except AttributeError:
            raise LookupError("Can't find information to return a proper time")

    def wavefront_sensor(self):
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

    def wavelength_band(self):
        # TODO: Make sure we get this in micrometers...
        ctrl_wave = self.effective_wavelength()

        def wavelength_diff((_, l)):
            return abs(l - ctrl_wave)
        band = min(wavelength_band.items(), key = wavelength_diff)[0]

        # TODO: This can't happen. We probably want to check against "None"
        if band is None:
            raise ValueError()

        return band

    def wcs_ra(self):
        raise NotImplementedError("wcs_dec needs types/tags, and direct access to header...")

    def wcs_dec(self):
        raise NotImplementedError("wcs_dec needs types/tags, and direct access to header...")

    ra = wcs_ra
    dec = wcs_dec
