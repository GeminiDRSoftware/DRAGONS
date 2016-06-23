import re
import datetime
import dateutil

# from gempy.gemini import gemini_metadata_utils as gmu
from astrodata import factory, AstroDataFits, simple_descriptor_mapping, keyword

# NOTE: Temporary functions for test. gempy imports astrodata and
#       won't work with this implementation
def removeComponentID(instr):
    """
    Remove a component ID from a filter name
    :param instr: the filter name
    :type instr: string
    :rtype: string
    :return: the filter name with the component ID removed
    """
    m = re.match (r"(?P<filt>.*?)_G(.*?)", instr)
    if not m:
                # There was no "_G" in the input string. Return the input string
        ret_str = str(instr)
    else:
                ret_str = str(m.group("filt"))
    return ret_str

def sectionStrToIntList(section):
    """
    Convert the input section in the form '[x1:x2,y1:y2]' to a list in the
    form [x1 - 1, x2, y1 - 1, y2], where x1, x2, y1 and y2 are
    integers. The values in the output list are converted to use 0-based and 
    non-inclusive indexing, making it compatible with numpy.

    :param section: the section (in the form [x1:x2,y1:y2]) to be
                    converted to a list
    :type section: string

    :rtype: list
    :return: the converted section as a list that uses 0-based and
             non-inclusive in the form [x1 - 1, x2, y1 - 1, y2]
    """
    # Strip the square brackets from the input section and then create a
    # list in the form ['x1:x2','y1:y2']
    xylist = section.strip('[]').split(',')

    # Create variables containing the single x1, x2, y1 and y2 values
    x1 = int(xylist[0].split(':')[0]) - 1
    x2 = int(xylist[0].split(':')[1])
    y1 = int(xylist[1].split(':')[0]) - 1
    y2 = int(xylist[1].split(':')[1])

    # Return the list in the form [x1 - 1, x2, y1 - 1, y2]
    return [x1, x2, y1, y2]

def parse_percentile(string):
        # Given the type of string that ought to be present in the site condition
    # headers, this function returns the integer percentile number
    #
    # Is it 'Any' - ie 100th percentile?
    if(string == "Any"):
        return 100

    # Is it a xx-percentile string?
    m = re.match("^(\d\d)-percentile$", string)
    if(m):
        return int(m.group(1))

    # We didn't recognise it
    return None

### END temporaty functions

# Gives the effective wavelength in microns for the standard wavelength regimes
wavelength_band = {
    "None" : 0.000,
    "u": 0.350,
    "g": 0.475,
    "r": 0.630,
    "i": 0.780,
    "Z": 0.900,
    "Y": 1.020,
    "X": 1.100,
    "J": 1.200,
    "H": 1.650,
    "K": 2.200,
    "L": 3.400,
    "M": 4.800,
    "N": 11.70,
    "Q": 18.30,
}

nominal_extinction = {
    # These are the nominal MK and CP extinction values
    # ie the k values where the magnitude of the star should be modified by 
    # -= k(airmass-1.0)
    #
    # Columns are given as:
    # (telescope, filter) : k
    #
    ('Gemini-North', 'u'): 0.42,
    ('Gemini-North', 'g'): 0.14,
    ('Gemini-North', 'r'): 0.11,
    ('Gemini-North', 'i'): 0.10,
    ('Gemini-North', 'z'): 0.05,
    #
    ('Gemini-South', 'u'): 0.38,
    ('Gemini-South', 'g'): 0.18,
    ('Gemini-South', 'r'): 0.10,
    ('Gemini-South', 'i'): 0.08,
    ('Gemini-South', 'z'): 0.05
}

# TODO: This should be moved to the instruments. Figure out a way...

# Instrument, filter effective wavlengths
# GMOS-N and GMOS-S are converted to GMOS
# '*' can be overriden by a specific instrument
filterWavelengths = {
    ('*',     'u')            : 0.3500,
    ('*',     'g')            : 0.4750,
    ('*',     'r')            : 0.6300,
    ('*',     'i')            : 0.7800,
    ('*',     'z')            : 0.9250,
    ('*',     'Y')            : 1.0200,
    ('*',     'J')            : 1.2500,
    ('*',     'H')            : 1.6350,
    ('*',     'K')            : 2.2000,
    ('GMOS',  'HeII')         : 0.4680,
    ('GMOS',  'HeIIC')        : 0.4780,
    ('GMOS',  'OIII')         : 0.4990,
    ('GMOS',  'OIIIC')        : 0.5140,
    ('GMOS',  'Ha')           : 0.6560,
    ('GMOS',  'HaC')          : 0.6620,
    ('GMOS',  'SII')          : 0.6720,
    ('GMOS',  'CaT')          : 0.8600,
    ('GMOS',  'Z')            : 0.8760,
    ('GMOS',  'DS920')        : 0.9200,
    ('GMOS',  'Y')            : 1.0100,
    ('GSAOI', 'Z')            : 1.0150,
    ('GNIRS', 'YPHOT')        : 1.0300,
    ('NIRI',  'Jcon(1065)')   : 1.0650,
    ('NIRI',  'HeI')          : 1.0830,
    ('GSAOI', 'HeI1083')      : 1.0830,
    ('NIRI',  'Pa(gamma)')    : 1.0940,
    ('GSAOI', 'PaG')          : 1.0940,
    ('GNIRS', 'X')            : 1.1000,
    ('GNIRS', 'X_(order_6)')  : 1.1000,
    ('NIRI',  'Jcon(112)')    : 1.1220,
    ('F2',    'Jlow')         : 1.1220,
    ('NIRI',  'Jcon(121)')    : 1.2070,
    ('GSAOI', 'Jcont')        : 1.2070,
    ('GNIRS', 'JPHOT')        : 1.2500,
    ('GNIRS', 'J_(order_5)')  : 1.2700,
    ('NIRI',  'Pa(beta)')     : 1.2820,
    ('GSAOI', 'PaB')          : 1.2820,
    ('F2'     'JH')           : 1.3900,
    ('NIRI',  'H-con(157)')   : 1.5700,
    ('GSAOI', 'Hcont')        : 1.5700,
    ('NIRI',  'CH4(short)')   : 1.5800,
    ('GSAOI', 'CH4short')     : 1.5800,
    ('GNIRS', 'H')            : 1.6300,
    ('GNIRS', 'H_(order_4)')  : 1.6300,
    ('F2',    'H')            : 1.6310,
    ('*',     'FeII')         : 1.6440,
    ('NIRI',  'H')            : 1.6500,
    ('NIRI',  'CH4(long)')    : 1.6900,
    ('GSAOI', 'CH4long')      : 1.6900,
    ('F2',    'HK')           : 1.8710,
    ('GSAOI', 'H2O')          : 2.0000,
    ('NIRI',  'H2Oice(2045)') : 2.0450,
    ('GSAOI', 'HeI-2p2s')     : 2.0580,
    ('NIRI',  'HeI(2p2s)')    : 2.0590,
    ('GSAOI', 'Kcntshrt')     : 2.0930,
    ('NIRI',  'Kcon(209)')    : 2.0975,
    ('NIRI',  'K(prime)')     : 2.1200,
    ('GSAOI', 'Kprime')       : 2.1200,
    ('GSAOI', 'H2(1-0)')      : 2.1220,
    ('NIRI',  'H2 1-0 S1')    : 2.1239,
    ('GNIRS', 'H2')           : 2.1250,
    ('NIRI',  'K(short)')     : 2.1500,
    ('GSAOI', 'Kshort')       : 2.1500,
    ('F2',    'Ks')           : 2.1570,
    ('NIRI',  'H2 2-1 S1')    : 2.2465,
    ('GSAOI', 'H2(2-1)')      : 2.2480,
    ('GSAOI', 'BrG')          : 2.1660,
    ('NIRI',  'Br(gamma)')    : 2.1686,
    ('GNIRS', 'K_(order_3)')  : 2.1950,
    ('F2',    'Klong')        : 2.2000,
    ('GNIRS', 'KPHOT')        : 2.2200,
    ('GSAOI', 'Kcntlong')     : 2.2700,
    ('NIRI',  'Kcon(227)')    : 2.2718,
    ('NIRI',  'CH4ice(2275)') : 2.2750,
    ('NIRI',  'CO 2-0 (bh)')  : 2.2890,
    ('GSAOI', 'CO2360')       : 2.3600,
    ('NIRI',  'H2Oice')       : 3.0500,
    ('NIRI',  'hydrocarb')    : 3.2950,
    ('GNIRS', 'PAH')          : 3.2950,
    ('GNIRS', 'L')            : 3.5000,
    ('GNIRS', 'L_(order_2)')  : 3.5000,
    ('NIRI',  'L(prime)')     : 3.7800,
    ('NIRI',  'Br(alpha)Con') : 3.9900,
    ('NIRI',  'Br(alpha)')    : 4.0520,
    ('NIRI',  'M(prime)')     : 4.6800,
    ('GNIRS', 'M')            : 5.1000,
    ('GNIRS', 'M_(order_1)')  : 5.1000,
}

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
    dec = keyword("DEC"),
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
    observation_epoch = keyword("OBSEPOCH"),
    observation_id = keyword("OBSID"),
    observation_type = keyword("OBSTYPE"),
    overscan_section = keyword("OVERSSEC"),
    pixel_scale = keyword("PIXSCALE"),
    prism = keyword("PRISM"),
    program_id = keyword("GEMPRGID"),
    pupil_mask = keyword("PUPILMSK"),
    r_zero_val = keyword("RZEROVAL"),
    ra = keyword("RA"),
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

    def _some_section(self, descriptor_name, keyword, pretty):
        try:
            value_filter = (str if pretty else sectionStrToIntList)
            return [(None if raw is None else value_filter(raw))
                    for raw in getattr(self.ext, keyword)]
        except KeyError:
            raise AttributeError("No {} information".format(descriptor_name))

    def _may_remove_component(self, keyword, stripID, pretty):
        val = getattr(self.phu, keyword)
        if stripID or pretty:
            return removeComponentID(val)
        return val

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
        raise NotImplementedError("dispersion_axis needs types/tags...")

    def effective_wavelength(self):
        raise NotImplementedError("effective_wavelength needs types/tags...")

    def exposure_time(self):
        # Keyword: exposure_time = keyword("EXPTIME"),
        raise NotImplementedError("effective_wavelength needs types/tags...")

    def filter_name(self, **kw):
        raise NotImplementedError("filter_name seems to be reimplemented by every instrument. Needs further investigation.")

    def focal_plane_mask(self, stripID=False, pretty=False):
        self._may_remove_component('FPMASK', stripID, pretty)

    def gcal_lamp(self):
        lamps, shut = self.phu.GCALLAMP, self.phu.GCALSHUT
        if (shut.upper() == 'CLOSED' and lamps.upper() in ('IRHIGH', 'IRLOW')) or lamps.upper() in ('', 'NO VALUE'):
            return 'Off'

        return lamps

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
        coeff = nominal_exctinction.get(nom_ext_idx, 0.0)

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

    def raw_bg(self):
        return _raw_to_percentile('raw_bg', self.phu.RAWBG)

    def raw_cc(self):
        return _raw_to_percentile('raw_cc', self.phu.RAWCC)

    def raw_iq(self):
        return _raw_to_percentile('raw_iq', self.phu.RAWIQ)

    def raw_wv(self):
        return _raw_to_percentile('raw_wv', self.phu.RAWWV)

    def requested_bg(self):
        return _requested_to_percentile('raw_bg', self.phu.REQBG)

    def requested_cc(self):
        return _requested_to_percentile('raw_cc', self.phu.REQCC)

    def requested_iq(self):
        return _requested_to_percentile('raw_iq', self.phu.REQIQ)

    def requested_wv(self):
        return _requested_to_percentile('raw_wv', self.phu.REQWV)

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
        raise NotImplementedError("wcs_ra needs types/tags...")

    def wcs_dec(self):
        raise NotImplementedError("wcs_dec needs types/tags...")

    ra = wcs_ra
    dec = wcs_dec


factory.addClass(AstroDataGemini)
