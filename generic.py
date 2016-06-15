# from gempy.gemini import gemini_metadata_utils as gmu
from astrodata import AstroDataFits, descriptor_keyword_mapping
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

gemini_direct_keywords = {
    "ao_fold":  "AOFOLD",
    "ao_seeing":  "AOSEEING",
    "aowfs": "AOWFS_ST",
    "array_name": "ARRAYNAM",
    "azimuth": "AZIMUTH",
    "bias_image": "BIASIM",
    "bunit": "BUNIT",
    "cd11": "CD1_1",
    "cd12": "CD1_2",
    "cd21": "CD2_1",
    "cd22": "CD2_2",
    "dark_image": "DARKIM",
    "data_label": "DATALAB",
    "dec": "DEC",
    "decker": "DECKER",
    "detector_name": "DETNAME",
    "detector_roi_setting": "DROISET",
    "detector_rois_requested": "DROIREQ",
    "detector_section": "DETSEC",
    "detector_x_bin": "XCCDBIN",
    "detector_y_bin": "YCCDBIN",
    "disperser": "DISPERSR",
    "dispersion": "WDELTA",
    "dispersion_axis": "DISPAXIS",
    "elevation": "ELEVATIO",
    "exposure_time": "EXPTIME",
    "filter_name": "FILTNAME",
    "focal_plane_mask": "FPMASK",
    "gain": "GAIN",
    "gain_setting": "GAINSET",
    "gems": "GWFS1CFG",
    "grating": "GRATING",
    "group_id": "GROUPID",
    "local_time": "LT",
    "lyot_stop": "LYOTSTOP",
    "mdf_row_id": "MDFROW",
    "naxis1": "NAXIS1",
    "naxis2": "NAXIS2",
    "nod_count": "NODCOUNT",
    "nod_pixels": "NODPIX",
    "nominal_atmospheric_extinction": "NOMATMOS",
    "nominal_photometric_zeropoint": "NOMPHOTZ",
    "non_linear_level": "NONLINEA",
    "observation_class": "OBSCLASS",
    "observation_epoch": "OBSEPOCH",
    "observation_id": "OBSID",
    "observation_type": "OBSTYPE",
    "oiwfs": "OIWFS_ST",
    "overscan_section": "OVERSSEC",
    "pixel_scale": "PIXSCALE",
    "prism": "PRISM",
    "program_id": "GEMPRGID",
    "pupil_mask": "PUPILMSK",
    "pwfs1": "PWFS1_ST",
    "pwfs2": "PWFS2_ST",
    "qa_state": "QASTATE",
    "r_zero_val":  "RZEROVAL",
    "ra": "RA",
    "raw_bg": "RAWBG",
    "raw_cc": "RAWCC",
    "raw_iq": "RAWIQ",
    "raw_wv": "RAWWV",
    "raw_gemini_qa": "RAWGEMQA",
    "raw_pi_requirements_met": "RAWPIREQ",
    "read_mode": "READMODE",
    "read_noise": "RDNOISE",
    "read_speed_setting": "RDSPDSET",
    "requested_bg": "REQBG",
    "requested_cc": "REQCC",
    "requested_iq": "REQIQ",
    "requested_wv": "REQWV",
    "saturation_level": "SATLEVEL",
    "slit": "SLIT",
    "ut_datetime": "DATETIME",
    "ut_time": "UT",
    "wavefront_sensor": "WFS",
    "wavelength":  "WAVELENG",
    "wavelength_band": "WAVEBAND",
    "wavelength_reference_pixel": "WREFPIX",
    "well_depth_setting": "WELDEPTH",
    "x_offset": "XOFFSET",
    "y_offset": "YOFFSET",
}

@descriptor_keyword_mapping(**gemini_direct_keywords)
class AstroDataGemini(AstroDataFits):
    @staticmethod
    def _matches_data(data_provider):
        obs = data_provider.header[0].get('OBSERVAT').upper()
        # This covers variants like 'Gemini-North', 'Gemini North', etc.
        return obs in ('GEMINI-NORTH', 'GEMINI-SOUTH')

    def airmass(self):
        am = self.keyword.AIRMASS

        if am < 1:
            raise ValueError("Can't have less than 1 airmass!")

        return float(am)

    def ao_seeing(self):
        try:
            return self.ao_seeing
        except KeyError:
            try:
                # If the r_zero_val keyword (Fried's parameter) is present, 
                # a seeing estimate can be calculated (NOTE: Jo Thomas-Osip 
                # is providing a reference for this calculation. Until then, 
                # EJD checked using 
                # http://www.ctio.noao.edu/~atokovin/tutorial/part1/turb.html )

                # Seeing at 0.5 micron
                rzv = self.r_zero_val
                return (206265. * 0.98 * 0.5e-6) / (rzv * 0.01)
            except KeyError:
                raise AttributeError("There is no information about AO seeing")

    def array_section(self, pretty=False):
        try:
            value_filter = (str if pretty else sectionStrToIntList)
            return dict(
                    (ext, (None if raw is None else value_filter(raw)))
                    for ext, raw in self.keyword.get_all('ARRAYSEC').items()
                   )
        except KeyError:
            raise AttributeError("No array_section information")

    def camera(self, stripID=False, pretty=False):
        camera = self.keyword.CAMERA
        if stripID or pretty:
            camera = removeComponentID(camera)

        return camera

    def cass_rotator_pa(self):
        val = float(self.keyword.CRPA)
        if val < -360 or val > 360:
            raise ValueError("Invalid CRPA value: {}".format(val))
        return val

    def central_wavelength(self):
        val = self.keyword.CWAVE
        if val < 0:
            raise ValueError("Invalid CWAVE value: {}".format(val))

        return val

    def coadds(self):
        return int(self.keyword.get('COADDS', 1))

    def data_section(self, pretty=False):
        try:
            value_filter = (str if pretty else sectionStrToIntList)
            return dict(
                    (ext, (None if raw is None else value_filter(raw)))
                    for ext, raw in self.keyword.get_all('DATASEC').items()
                   )
        except KeyError:
            raise AttributeError("No data_section information")
