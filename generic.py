# from gempy.gemini import gemini_metadata_utils as gmu
from astrodata import factory, AstroDataFits, simple_descriptor_mapping, keyword
import re

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

gemini_direct_keywords = dict(
    ao_fold = keyword("AOFOLD"),
    aowfs = keyword("AOWFS_ST"),
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
    decker = keyword("DECKER"),
    detector_name = keyword("DETNAME"),
    detector_roi_setting = keyword("DROISET"),
    detector_rois_requested = keyword("DROIREQ"),
    detector_section = keyword("DETSEC"),
    detector_x_bin = keyword("XCCDBIN"),
    detector_y_bin = keyword("YCCDBIN"),
    disperser = keyword("DISPERSR"),
    dispersion = keyword("WDELTA"),
    dispersion_axis = keyword("DISPAXIS"),
    elevation = keyword("ELEVATIO"),
    exposure_time = keyword("EXPTIME"),
    filter_name = keyword("FILTNAME"),
    focal_plane_mask = keyword("FPMASK"),
    gain = keyword("GAIN"),
    gain_setting = keyword("GAINSET"),
    gems = keyword("GWFS1CFG"),
    grating = keyword("GRATING"),
    group_id = keyword("GROUPID"),
    local_time = keyword("LT"),
    lyot_stop = keyword("LYOTSTOP"),
    mdf_row_id = keyword("MDFROW"),
    naxis1 = keyword("NAXIS1"),
    naxis2 = keyword("NAXIS2"),
    nod_count = keyword("NODCOUNT"),
    nod_pixels = keyword("NODPIX"),
    nominal_atmospheric_extinction = keyword("NOMATMOS"),
    nominal_photometric_zeropoint = keyword("NOMPHOTZ"),
    non_linear_level = keyword("NONLINEA"),
    observation_class = keyword("OBSCLASS"),
    observation_epoch = keyword("OBSEPOCH"),
    observation_id = keyword("OBSID"),
    observation_type = keyword("OBSTYPE"),
    oiwfs = keyword("OIWFS_ST"),
    overscan_section = keyword("OVERSSEC"),
    pixel_scale = keyword("PIXSCALE"),
    prism = keyword("PRISM"),
    program_id = keyword("GEMPRGID"),
    pupil_mask = keyword("PUPILMSK"),
    pwfs1 = keyword("PWFS1_ST"),
    pwfs2 = keyword("PWFS2_ST"),
    qa_state = keyword("QASTATE"),
    r_zero_val = keyword("RZEROVAL"),
    ra = keyword("RA"),
    raw_bg = keyword("RAWBG"),
    raw_cc = keyword("RAWCC"),
    raw_iq = keyword("RAWIQ"),
    raw_wv = keyword("RAWWV"),
    raw_gemini_qa = keyword("RAWGEMQA"),
    raw_pi_requirements_met = keyword("RAWPIREQ"),
    read_mode = keyword("READMODE"),
    read_noise = keyword("RDNOISE"),
    read_speed_setting = keyword("RDSPDSET"),
    requested_bg = keyword("REQBG"),
    requested_cc = keyword("REQCC"),
    requested_iq = keyword("REQIQ"),
    requested_wv = keyword("REQWV"),
    saturation_level = keyword("SATLEVEL"),
    slit = keyword("SLIT"),
    ut_datetime = keyword("DATETIME"),
    ut_time = keyword("UT"),
    wavefront_sensor = keyword("WFS"),
    wavelength = keyword("WAVELENG"),
    wavelength_band = keyword("WAVEBAND"),
    wavelength_reference_pixel = keyword("WREFPIX"),
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
        try:
            value_filter = (str if pretty else sectionStrToIntList)
            return [(None if raw is None else value_filter(raw))
                    for raw in self.ext().ARRAYSEC]
        except KeyError:
            raise AttributeError("No array_section information")

    def camera(self, stripID=False, pretty=False):
        camera = self.phu.CAMERA
        if stripID or pretty:
            camera = removeComponentID(camera)

        return camera

    def cass_rotator_pa(self):
        val = float(self.phu.CRPA)
        if val < -360 or val > 360:
            raise ValueError("Invalid CRPA value: {}".format(val))
        return val

    def central_wavelength(self):
        val = self.phu.CWAVE
        if val < 0:
            raise ValueError("Invalid CWAVE value: {}".format(val))

        return val

    def coadds(self):
        return int(self.phu.get('COADDS', 1))

    def data_section(self, pretty=False):
        try:
            value_filter = (str if pretty else sectionStrToIntList)
            return [(None if raw is None else value_filter(raw))
                    for raw in self.ext().DATASEC]
        except KeyError:
            raise AttributeError("No data_section information")

factory.addClass(AstroDataGemini)
