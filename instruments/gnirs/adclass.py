import math

from astrodata import astro_data_tag, simple_descriptor_mapping, keyword
from ..gemini import AstroDataGemini
from .lookups import detector_properties, nominal_zeropoints, config_dict, read_modes

# NOTE: Temporary functions for test. gempy imports astrodata and
#       won't work with this implementation
from ..gmu import *

@simple_descriptor_mapping(
    bias = keyword("DETBIAS"),
    central_wavelength = keyword("GRATWAVE"),
    filter1 = keyword("FILTER1"),
    filter2 = keyword("FILTER2"),
    hicol = keyword("HICOL", on_ext=True),
    hirow = keyword("HIROW", on_ext=True),
    lnrs = keyword("LNRS"),
    lowcol = keyword("LOWCOL", on_ext=True),
    lowrow = keyword("LOWROW", on_ext=True),
    ndavgs = keyword("NDAVGS")
)
class AstroDataGnirs(AstroDataGemini):
    @staticmethod
    def _matches_data(data_provider):
        return data_provider.phu.get('INSTRUME').upper() == 'GNIRS'

    @astro_data_tag
    def _type_dark(self):
        if self.phu.OBSTYPE == 'DARK':
            return (set(['DARK', 'CAL']), set(['IMAGE', 'SPECT']))

    @astro_data_tag
    def _type_image(self):
        # NOTE: Prevented by DARK
        if self.phu.ACQMIR == 'In':
            tags = set(['IMAGE'])
            if self.phu.get('OBSTYPE') == 'FLAT':
                tags.add('FLAT')
                tags.add('CAL')
            return (tags, ())

    @astro_data_tag
    def _type_spect(self):
        # NOTE: Prevented by DARK
        if self.phu.ACQMIR == 'Out':
            tags = set(['SPECT'])
            slit = self.phu.get('SLIT', '')
            if 'arcsec' in slit:
                tags.add('LS')
            elif slit == 'IFU':
                tags.add('IFU')
            return (tags, ())

    @astro_data_tag
    def _type_pinhole(self):
        if self.phu.OBSTYPE == 'FLAT':
            if self.phu.SLIT in ('LgPinholes_G5530', 'SmPinholes_G5530'):
                return (set(['PINHOLE', 'CAL']), ())

    def data_section(self, pretty=False):
        hirows = self.hirow()
        lowrows = self.lowrow()
        hicols = self.hicol()
        lowcols = self.lowcol()

        data_sections = []
        # NOTE: Rows are X and cols are Y? These Romans are crazy
        for hir, hic, lowr, lowc in zip(hirows, hicols, lowrows, locols):
            if pretty:
                item = "[{:d}:{:d},{:d}:{:d}]".format(lowr+1, hir+1, lowc+1, hic+1)
            else:
                item = (lowr, hir+1, lowc, hic+1)
            data_sections.append(item)

        return data_sections

    array_section = data_section
    detector_section = data_section

    def disperser(self, stripID=False, pretty=False):
        if self.phu.get('ACQMIR') == 'In':
            return 'MIRROR'

        grating = self.grating(stripID=stripID, pretty=pretty)
        prism = self.prism(stripID=stripID, pretty=pretty)
        if prism.startswith('MIR'):
            return str(grating)
        else:
            return "{}&{}".format(grating, prism)

    def focal_plane_mask(self, stripID=False, pretty=False):
        slit = self.slit(stripID=stripID, pretty=pretty).replace('Acquisition', 'Acq')
        decker = self.decker(stripID=stripID, pretty=pretty).replace('Acquisition', 'Acq')

        # Default fpm value
        fpm = "{}&{}".format(slit, decker)
        if pretty:
            if "Long" in decker:
                fpm = slit
            elif "XD" in decker:
                fpm = "{}XD".format(slit)
            elif "IFU" in slit and "IFU" in decker:
                fpm = "IFU"
            elif "Acq" in slit and "Acq" in decker:
                fpm = "Acq"

        return fpm

    def gain(self):
        read_mode = self.read_mode()
        well_depth = self.well_depth_setting()

        return float(detector_properties[read_mode, well_depth].gain)

    def grating(self, stripID=False, pretty=False):
        grating = self.phu.GRATING

        match = re.match("([\d/m]+)([A-Z]*)_G(\d+)", grating)
        try:
            ret_grating = "{}{}{}".format(*match.groups())
        except AttributeError:
            ret_grating = grating

        if stripID or pretty:
            return removeComponentID(ret_grating)
        return ret_grating

    def group_id(self):
        # For GNIRS image data, the group id contains the read_mode,
        # well_depth_setting, detector_section.
        # In addition flats, twilights and camera have the pretty version of the
        # filter_name included. For science data the pretty version of the
        # observation_id, filter_name and the camera are also included.

        tags = self.tags()
        if 'DARK' in tags:
            desc_list = 'read_mode', 'exposure_time', 'coadds'
        else:
            # The descriptor list is the same for flats and science frames
            desc_list = 'observation_id', 'filter_name', 'camera', 'read_mode'

        desc_list = desc_list + ('well_depth_setting', 'detector_section', 'disperser', 'focal_plane_mask')

        pretty_ones = set(['filter_name', 'disperser', 'focal_plane_mask'])

        collected_strings = []
        for desc in desc_list:
            method = getattr(self, desc)
            if desc in pretty_ones:
                result = method(pretty=True)
            else:
                result = method()
            collected_strings.append(str(result))

        return '_'.join(collected_strings)

    def nominal_photometric_zeropoint(self):
        gain = self.gain()
        camera = self.camera()
        filter_name = self.filter_name(pretty=True)

        result = []
        for bunit in self.ext.BUNIT:
            gain_factor = (2.5 * math.log10(gain)) if bunit == 'adu' else 0.0
            nz_key = (filter_name, camera)
            nom_phot_zeropoint = nominal_zeropoints[nz_key] - gain_factor
            result.append(nom_phot_zeropoint)

        return result

    def non_linear_level(self):
        read_mode = self.read_mode()
        well_depth = self.well_depth_setting()

        limit = detector_properties[read_mode, well_depth].linearlimit

        return int(limit * self.saturation_level())

    def pixel_scale(self):
        camera = self.camera()

        if self.tags() & set(['IMAGE', 'DARK']):
            # Imaging or darks
            match = re.match("^(Short|Long)(Red|Blue)_G\d+$", camera)
            try:
                cameratype = match.group(1)
                if cameratype == 'Short':
                    ret_pixel_scale = 0.15
                elif cameratype == 'Long':
                    ret_pixel_scale = 0.05
            except AttributeError:
                raise Exception('No camera match for imaging mode')
        else:
            # Spectroscopy mode
            prism = self.phu.PRISM
            decker = self.phu.DECKER
            disperser = self.phu.GRATING

            ps_key = (prism, decker, disperser, camera)
            ret_pixel_scale = float(config_dict[ps_key].pixscale)

        return ret_pixel_scale

    def prism(self, stripID=False, pretty=False):
        prism = self.phu.PRISM
        match = re.match("[LBSR]*\+*([A-Z]*_G\d+)", prism)
        # NOTE: The original descriptor has no provisions for not matching
        #       the RE... which will produce an exception down the road.
        #       Let's do it here (it will be an AttributeError, though)
        ret_prism = match.group(2)

        if stripID or pretty:
            ret_prism = removeComponentID(ret_prism)

        return ret_prism

    def ra(self):
        wcs_ra = self.wcs_ra()
        tgt_ra = self.target_ra(offset=True, icrs=True)
        delta = abs(wcs_ra - tgt_ra)

        # wraparound?
        if delta > 180:
            delta = abs(delta - 360)
        delta = delta * 3600 # to arcsecs

        # And account for cos(dec) factor
        delta /= math.cos(math.radians(self.dec()))

        # If more than 1000" arcsec different, WCS is probably bad
        return (tgt_ra if delta > 1000 else wcs_ra)

    def dec(self):
        # In general, the GNIRS WCS is the way to go. But sometimes the DC
        # has a bit of a senior moment and the WCS is miles off (presumably
        # still has values from the previous observation or something. Who knows.
        # So we do a sanity check on it and use the target values if it's messed up
        wcs_dec = self.wcs_dec()
        tgt_dec = self.target_dec(offset=True, icrs=True)

        # wraparound?
        if delta > 180:
            delta = abs(delta - 360)
        delta = delta * 3600 # to arcsecs

        # If more than 1000" arcsec different, WCS is probably bad
        return (tgt_dec if delta > 1000 else wcs_dec)

    def read_mode(self):
        # Determine the number of non-destructive read pairs (lnrs) and the
        # number of digital averages (ndavgs) keywords from the global keyword
        # dictionary
        lnrs = self.lnrs()
        ndavgs = self.ndavgs()

        return read_modes.get((lnrs, ndavgs), "Invalid")

    def read_noise(self):
        # Determine the read mode and well depth from their descriptors
        read_mode = self.read_mode()
        well_depth = self.well_depth_setting()
        coadds = self.coadds()

        read_noise = detector_properties[(read_mode, well_depth)].readnoise

        return read_noise * math.sqrt(coadds)

    def saturation_level(self):
        gain = self.gain()
        coadds = self.coadds()
        read_mode = self.read_mode()
        well_depth = self.well_depth_setting()
        well = detector_properties[(read_mode, well_depth)].well

        return int(well * coadds / gain)

    def slit(self, stripID=False, pretty=False):
        slit = self.phu.SLIT.replace(' ', '')

        return (removeComponentID(slit) if stripID or pretty else slit)

    def well_depth_setting(self):
        biasvolt = self.bias()

        if abs(0.3 - biasvolt) < 0.1:
            return "Shallow"
        elif abs(0.6 - biasvolt) < 0.1:
            return "Deep"
        else:
            return "Invalid"
