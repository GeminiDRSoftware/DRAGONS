# This code follows the Gemini standard for other instruments
# even though pylint doesn't like it.
# pylint: disable=no-self-use, inconsistent-return-statements

"""
This module contains the AstroDataGhost class, used for adding tags
and descriptors to GHOST data.
"""

from astrodata import (
    astro_data_tag,
    TagSet,
    astro_data_descriptor,
)
from gemini_instruments.gemini import AstroDataGemini
from gemini_instruments.common import build_group_id


def return_dict_for_bundle(desc_fn):
    """
    A decorator that will return a dict with keys "blue" and "red" and values
    equal to the descriptor return if it was sent the split files. A check is
    made that all the returns are equal, and None is returned otherwise.

    This works by splitting the bundle and evaluating the descriptor on all of
    the blue and then red arms in turn. It therefore works regardless of whether
    the required information is in the extensions or the nascent PHU.

    Its behaviour on multi-extension slices is unclear.
    """
    def wrapper(self, *args, **kwargs):
        if not self.is_single and 'BUNDLE' in self.tags:
            ret_dict = dict()
            cameras = self.hdr.get('CAMERA', '')
            namps = self.hdr.get('NAMPS')
            i = 0
            while i < len(namps):
                camera = cameras[i].lower()
                # We've found the nascent PHU of an arm exposure
                if namps[i] > 1:
                    ret_value = desc_fn(self[i+1:i+namps[i]+1], *args, **kwargs)
                    i += namps[i] + 1
                else:  # it's a slitviewer
                    ret_value = desc_fn(self[i], *args, **kwargs)
                    i += 1
                # Check for single-valuedness of all returns from this camera
                try:
                    if ret_dict[camera] != ret_value:
                        ret_dict[camera] = None
                except KeyError:
                    ret_dict[camera] = ret_value
            return ret_dict
        return desc_fn(self, *args, **kwargs)
    return wrapper


def use_nascent_phu_for_bundle(desc_fn):
    """
    A decorator for bundles (where the PHU is minimal) that will instead
    provide the first nascent PHU (of a red/blue arm) instead
    """
    def wrapper(self, *args, **kwargs):
        if 'BUNDLE' in self.tags:
            phu_index = min(i for i, camera in enumerate(self.hdr['CAMERA'])
                            if camera in ('BLUE', 'RED'))
            nascent_phu = self[phu_index]
            return desc_fn(nascent_phu, *args, **kwargs)
        return desc_fn(self, *args, **kwargs)
    return wrapper


class AstroDataGhost(AstroDataGemini):
    """
    Class for adding tags and descriptors to GHOST data.
    """

    __keyword_dict = dict(array_section = 'CCDSEC',
                          array_name = 'AMPNAME',
                          overscan_section = 'BIASSEC',
                          res_mode = 'SMPNAME',
                          exposure_time = 'EXPTIME',
                          saturation_level = 'SATURATE',
                          )

    def __iter__(self):
        if self.is_single:
            yield self
        else:
            for n in range(len(self)):
                yield self[n]

    def __getitem__(self, idx):
        """
        Override default slicing method for bundles for two reasons:
        1) Prevent creation of a new AD from non-contiguous extensions, as
           this doesn't make sense.
        2) Prevent creation of a new "Frankenstein" AD with data from more
           than one camera, as only original bundles should have this.
        """
        obj = super().__getitem__(idx)
        if 'BUNDLE' in self.tags:
            #print("BUNDLE", idx, obj._mapping)
            if max(obj.indices) - min(obj.indices) != len(obj.indices) - 1:
                raise ValueError("Bundles can only be sliced contiguously")
            #print(obj.tags)
            # Find the nascent PHU if it's not a slit viewer, keep PHU if it is
            # this copes with CAMERA return a list or a string (single-slice)
            if obj.hdr.get('CAMERA')[0].startswith('S'):
                # add in the first SLITV header so keywords are there
                # (which is explicitly done in debundling)
                objhdr = obj.hdr if obj.is_single else obj[0].hdr
                phu = self.phu + objhdr
                # because people just do things without understanding the ramifications
                phu['EXPTIME'] = objhdr['EXPTIME']
                obj.phu = phu
                return obj
            for i in range(min(obj.indices), -1, -1):
                ndd = self._all_nddatas[i]
                if not ndd.shape:
                    phu = ndd.meta['header']
                    break
            else:
                phu = self._all_nddatas[min(obj.indices)].meta['header']
            obj.phu = phu
            # This has to happen way down here because only by resetting the PHU
            # can we prevent the 'BUNDLE' tag appearing and hence allow more
            # slicing to occur
            if len(set(ext.shape for ext in obj)) > 1:
                raise ValueError("Bundles must be sliced from the same camera")
            #print("RETURNING", len(obj))
        return obj

    @astro_data_descriptor
    def data_label(self):
        """
        Returns the data label of an observation.

        Returns
        -------
        str
            the observation's data label
        """
        if super().data_label() is None:
            raise RuntimeError("Your data has no DATALAB header keyword")
        return super().data_label()

    @staticmethod
    def _matches_data(source):
        """
        Check if data is from GHOST.

        Parameters
        ----------
        source : astrodata.AstroData
            The source file to check.
        """
        return source[0].header.get('INSTRUME', '').upper() == 'GHOST'

    @astro_data_tag
    def _tag_instrument(self):
        """
        Define the minimal tag set for GHOST data.
        """
        return TagSet(['GHOST'])

    @astro_data_tag
    def _tag_bundle(self):
        """
        Define the 'bundled data' tag set for GHOST data.
        """
        # Gets blocked by tags created by split files
        return TagSet(['BUNDLE'])

    @astro_data_tag
    def _tag_bias(self):
        """
        Define the 'bias data' tag set for GHOST data.
        """
        if self.phu.get('OBSTYPE') == 'BIAS':
            return TagSet(['CAL', 'BIAS'], blocks=['IMAGE', 'SPECT'])

    @astro_data_tag
    def _tag_dark(self):
        """
        Define the 'dark data' tag set for GHOST data.
        """
        if self.phu.get('OBSTYPE') == 'DARK':
            return TagSet(['CAL', 'DARK'], blocks=['IMAGE', 'SPECT'])

    @astro_data_tag
    def _tag_arc(self):
        """
        Define the 'arc data' tag set for GHOST data.
        """
        if self.phu.get('OBSTYPE') == 'ARC':
            return TagSet(['CAL', 'ARC'])

    @astro_data_tag
    def _tag_flat(self):
        """
        Define the 'flat data' tag set for GHOST data.
        """
        if self.phu.get('OBSTYPE') == 'FLAT':
            return TagSet(['CAL', 'FLAT'])

    @astro_data_tag
    def _tag_slitflat(self):
        """
        Define the 'slitflat data' tag set for GHOST data.
        """
        if (self.phu.get('OBSTYPE') == 'FLAT' and
            self.phu.get('CAMERA', '').lower().startswith('slit')):
            return TagSet(['CAL', 'SLITFLAT'])

    @astro_data_tag
    def _tag_sky(self):
        """
        Define the 'flat data' tag set for GHOST data.
        """
        if self.phu.get('OBSTYPE') == 'SKY':
            return TagSet(['SKY'])

    # MCW 191107 - had to add SLIT back in to make cal system work
    @astro_data_tag
    def _tag_slitv(self):
        """
        Define the 'slit data' tag set for GHOST data.
        """
        if self.phu.get('CAMERA', '').lower().startswith('slit'):
            return TagSet(['SLITV', 'SLIT'],
                          blocks=['SPECT', 'BUNDLE'])

    @astro_data_tag
    def _tag_image(self):
        """
        Tag slitviewer images as IMAGE (must be done separately from
        the "SLITV" so "SLITV" isn't blocked by BIAS or BPM)
        """
        if self.phu.get('CAMERA', '').lower().startswith('slit'):
            return TagSet(['IMAGE'])

    @astro_data_tag
    def _tag_camera(self):
        """
        Define the 'spectrograph data' tag set for GHOST data.
        """
        # Also returns BLUE or RED if the CAMERA keyword is set thus
        camera = self.phu.get('CAMERA')
        if camera in ('BLUE', 'RED'):
            return TagSet([camera], blocks=['BUNDLE'])

    @astro_data_tag
    def _tag_spect(self):
        """
        Tag echelle frames as SPECT (must be done separately from
        "RED"/"BLUE" so that isn't blocked by BIAS or BPM)
        """
        camera = self.phu.get('CAMERA')
        if camera in ('BLUE', 'RED'):
            return TagSet(['SPECT', 'XD'], blocks=['BUNDLE'])

    @astro_data_tag
    def _status_processed_ghost_cals(self):
        """
        Define the 'processed data' tag set for GHOST data.
        """
        kwords = set(['PRSLITIM', 'PRSLITBI', 'PRSLITDA', 'PRSLITFL',
                      'PRWAVLFT', 'PRPOLYFT', 'PROCSTND'])
        if set(self.phu) & kwords:
            return TagSet(['PROCESSED'])

    @astro_data_tag
    def _tag_processed_standard(self):
        if 'PROCSTND' in self.phu:
            return TagSet(['STANDARD'])

    #@astro_data_tag
    #def _tag_binning_mode(self):
    #    """
    #    TODO: this should not be a tag
    #    Define the tagset for GHOST data of different binning modes.
    #    """
    #    binnings = self.hdr.get('CCDSUM')
    #    if binnings is None:  # CJS hack
    #        return TagSet([])
    #    if isinstance(binnings, list):
    #        binnings = [x for x in binnings if x]
    #        if all(x == binnings[0] for x in binnings):
    #            return TagSet([binnings[0].replace(' ', 'x', 1)])
    #        else:
    #            return TagSet(['NxN'])
    #    else:
    #        # A list should always be returned but it doesn't
    #        # hurt to be able to handle a string just in case
    #        return TagSet([binnings.replace(' ', 'x', 1)])

    @astro_data_tag
    def _tag_obsclass(self):
        """
        Define the tagset for 'partnerCal' observations.
        """
        if self.phu.get('OBSCLASS') == 'partnerCal':
            return TagSet(['PARTNER_CAL'])

    @astro_data_descriptor
    @return_dict_for_bundle
    def amp_read_area(self, pretty=False):
        """
        Returns a list of amplifier read areas, one per extension, made by
        combining the amplifier name and detector section; or, returns a
        string if called on a single-extension slice.

        Note: this descriptor is only used for calibration association
        purposes in the archive, not during DR, so for pragmatic reasons
        we will compress a list of identical elements to a single-element list
        to aid the archive code when matching SLITV calibrations (which will
        only have one extension)

        Returns
        -------
        list/str
            read_area of each extension
        """
        # Note that tiled arrays won't have an array_name, so we'll fake it
        # FIXME correctly fetch keyword for tileArrays primitive
        if self.phu.get('TILEARRY', None) is not None:
            ampname = [0, ]
        else:
            ampname = self.array_name()
        detsec = self.detector_section(pretty=True)
        # Combine the amp name(s) and detector section(s)
        if self.is_single:
            return "'{}':{}".format(ampname,
                        detsec) if ampname and detsec else None
        else:
            ret_value = ["'{}':{}".format(a,d)
                         if a is not None and d is not None else None
                         for a,d in zip(ampname, detsec)]
            if ret_value == ret_value[::-1]:
                return ret_value[:1]
            return ret_value

    @astro_data_descriptor
    def arm(self):
        """
        Returns a string indicating whether these data are from the red
        or blue arm of the spectrograph.

        Returns
        -------
        str/None
            Color of the arm (`'blue'`, `'red'`), or `'slitv'` in case of slit
            viewer data. Returns `None` if arm/slit status can't be determined.
        """
        tags = self.tags
        if 'BLUE' in tags:
            return 'blue'
        elif 'RED' in tags:
            return 'red'
        elif 'SLITV' in tags:
            return 'slitv'
        return None

    @astro_data_descriptor
    @return_dict_for_bundle
    def array_name(self):
        """
        Return the arr

        Returns
        -------
        str: a concatenated string of the detector name and amplifier
        """
        if self.is_single:
            return f"{self.detector_name()}, {self.hdr.get('AMPNAME')}"
        else:
            return [f"{ext.detector_name()}, {ext.hdr.get('AMPNAME')}" for ext in self]

    @astro_data_descriptor
    @return_dict_for_bundle
    def binning(self):
        """
        Returns an "MxN"-style string because CJS is fed up with not having this!
        """
        return super().binning()

    @astro_data_descriptor
    def calibration_key(self):
        """
        Returns a suitable calibration key for GHOST, which includes the arm.
        """
        return (self.data_label().replace('_stack', ''), self.arm())

    # FIXME Remove once headers corrected
    @astro_data_descriptor
    def central_wavelength(self, asMicrometers=False, asNanometers=False,
                           asAngstroms=False): # pragma: no cover
        """
        Dummy to work around current Gemini cal_mgr
        """

        val = self.phu.get(self._keyword_for('central_wavelength'), None)

        if val is None:
            if self.arm() == 'red':
                val = 4000. * 10**-10
            elif self.arm() == 'blue':
                val = 6000. * 10**-10
            else:
                return None


        if asMicrometers:
            val *= 10**6
        elif asNanometers:
            val *= 10**9
        elif asAngstroms:
            val *= 10**10

        return float(val)

    @astro_data_descriptor
    @return_dict_for_bundle
    def detector_name(self, pretty=False):
        """
        Returns the detector (CCD) name.
        """
        return self.phu.get('DETECTOR')

    @astro_data_descriptor
    @return_dict_for_bundle
    def detector_x_bin(self):
        """
        Returns the detector binning in the x-direction.

        Returns
        -------
        int
            The detector binning
        """
        def _get_xbin(binning):
            try:
                return int(binning.split()[0])
            except (AttributeError, ValueError):
                return None

        binning = self.hdr.get('CCDSUM')
        if self.is_single:
            return _get_xbin(binning)
        else:
            xbin_list = [_get_xbin(b) for b in binning]
            # Check list is single-valued
            return xbin_list[0] if xbin_list == xbin_list[::-1] else None

    @astro_data_descriptor
    @return_dict_for_bundle
    def detector_y_bin(self):
        """
        Returns the detector binning in the y-direction.

        Returns
        -------
        int
            The detector binning
        """
        def _get_ybin(binning):
            try:
                return int(binning.split()[1])
            except (AttributeError, ValueError, IndexError):
                return None

        binning = self.hdr.get('CCDSUM')
        if self.is_single:
            return _get_ybin(binning)
        else:
            ybin_list = [_get_ybin(b) for b in binning]
            # Check list is single-valued
            return ybin_list[0] if ybin_list == ybin_list[::-1] else None

    # TODO: GHOST descriptor returns no values if data are unprepared

    @astro_data_descriptor
    @return_dict_for_bundle
    def exposure_time(self):
        """
        Returns the exposure time. If run on a bundle, it returns the exposure
        time of a single exposure in each arm, NOT the total exposure time

        Returns
        -------
        int
            exposure time of a single exposure
        """
        return super().exposure_time()

        # Don't let this special logic happen for bundles
        #if 'BUNDLE' not in self.tags:
        #    if exp_time_default is None:
        #        exposure_time = self[0].hdr.get(
        #            self._keyword_for('exposure_time'),
        #            -1)
        #        if exposure_time == -1:
        #            return None
        #        return exposure_time
        #
        #return exp_time_default

    @astro_data_descriptor
    def focal_plane_mask(self, *args, **kwargs):
        """
        Returns the "focal plane mask", primarily to populate the archive's
        Header table so it can be searched on.

        Returns
        -------
        str
            "HR"/"SR" as appropriate
        """
        try:
            return self.res_mode().upper()[0]+"R"
        except AttributeError:
            return None

    # The gain() descriptor is inherited from gemini/adclass, and returns
    # the value of the GAIN keyword (as a list if sent a complete AD object,
    # or as a single value if sent a slice). This is what the GHOST version
    # does so one is not needed here.

    @astro_data_descriptor
    @return_dict_for_bundle
    def gain_setting(self):
        """
        Returns the gain setting for this observation (e.g., 'high', 'low')

        Returns
        -------
        str
            the gain setting
        """
        # TODO: confirm returns. Future-proofing for a possible high-gain mode
        if 'SLITV' in self.tags:
            return "standard"
        gain = self.gain()
        if self.is_single:
            return "low" if gain < 1.0 else "high"
        low_gain = [g <= 1.0 for g in gain]
        if all(low_gain):
            return "low"
        elif any(low_gain):
            raise ValueError("Some gains are low and some are high")
        return "high"

    @astro_data_descriptor
    def group_id(self):
        """
        Returns a string representing a group of data that are compatible
        with each other.  This is used when stacking, for example.  Each
        instrument, mode of observation, and data type will have its own rules.

        Returns
        -------
        str
            A group ID for compatible data
        """
        tags = self.tags
        if 'DARK' in tags:
            desc_list = ['exposure_time', 'coadds']
        elif 'BIAS' in tags:
            desc_list = []
        else:  # science exposures (and ARCs)
            desc_list = ['observation_id', 'res_mode']
        desc_list.append('arm')

        # never stack frames of mixed binning modes
        desc_list.append('detector_x_bin')
        desc_list.append('detector_y_bin')

        # MCW: We care about the resolution mode EXCEPT for dark and bias
        if 'DARK' not in tags and 'BIAS' not in tags:
            desc_list.append('res_mode')

        # CJS: Generally need to stop FLATs being stacked with science
        additional_item = 'FLAT' if 'FLAT' in tags else None

        return build_group_id(self, desc_list, prettify=[],
                              additional=additional_item)

    @astro_data_descriptor
    def non_linear_level(self):
        """
        Returns the level at which the data become non-linear. This is
        the same as the saturation level for the GHOST CCDs.

        Returns
        -------
        int/list
            non-linearity level
        """
        return self.saturation_level()

    @astro_data_descriptor
    def number_of_exposures(self):
        """
        Return the number of individual exposures

        Returns
        -------
        int/dict
            number of exposures
        """
        if 'BUNDLE' not in self.tags:
            return len(self) if 'SLITV' in self.tags else 1  # probably
        if self.is_single:
            return 1
        cameras = self.hdr.get('CAMERA')
        namps = self.hdr.get('NAMPS')
        ret_value = {'blue': 0, 'red': 0, 'slitv': 0}
        for camera, namp in zip(cameras, namps):
            if namp is not None:
                ret_value[camera.lower()] += 1
        return ret_value

    @astro_data_descriptor
    @return_dict_for_bundle
    def read_mode(self):
        """
        Returns a string describing the read mode, matching that offered
        in the OT. Only the readout speed will be configurable, so that
        is what is returned, albeit in a circuitous way to future-proof.

        Returns
        -------
        str
            The read mode
        """
        # TODO: get appropriate return values
        _read_mode_dict = {("slow", "low"): "slow",
                           ("medium", "low"): "medium",
                           ("fast", "low"): "fast",
                           ("fast", "high"): "bright",
                           ("standard", "standard"): "standard"}  # SLITV
        return _read_mode_dict.get((self.read_speed_setting(),
                                    self.gain_setting()), "unknown")

    # TODO: read_noise(): see comments on gain()

    @astro_data_descriptor
    @return_dict_for_bundle
    def read_speed_setting(self):
        """
        Returns the setting for the readout speed (slow or fast)

        Returns
        -------
        str
            The read speed ("slow"/"medium"/"fast")
        """
        if 'SLITV' in self.tags:
            return "standard"
        return ("slow", "medium", "fast", "unknown")[self.phu.get('READMODE', 3)]

    @astro_data_descriptor
    def res_mode(self):
        """
        Get the GHOST resolution mode of this dataset

        Returns
        -------
        str/None
            Resolution of the dataset ('high' | 'std'). Returns `None` if
            resolution mode cannot be determined.
        """
        mode = self.phu.get('SMPNAME')
        try:
            if mode.endswith('HI_ONLY'):
                return 'high'
            elif (mode.endswith('LO_ONLY') or mode.endswith('STD_ONLY')):
                return 'std'
        except Exception:
            pass
        return None

    @astro_data_descriptor
    def saturation_level(self):
        """Patch because SATURATE=0 for the blue spectrograph"""
        retval = super().saturation_level()
        if 'PREPARED' in self.tags:
            return retval
        if self.is_single:
            return None if retval is None else retval if retval > 0 else 65535
        return [None if v is None else v if v > 0 else 65535 for v in retval]

    @astro_data_descriptor
    @use_nascent_phu_for_bundle
    def ut_datetime(self, *args, **kwargs):
        return AstroDataGemini.ut_datetime(self, *args, **kwargs)

    @astro_data_descriptor
    def want_before_arc(self):
        """
        This is a special descriptor which is being used as a calibration
        system work-around. Outside of active reduction, this descriptor
        should always return None, as the relevant header keyword should only
        exist very briefly during the fetching of bracketed arc files.

        Returns
        -------
        bool or `None`
        """
        want_before = self.phu.get('ARCBEFOR', None)
        if want_before:
            return True
        elif want_before is None:
            return None
        else:
            return False
