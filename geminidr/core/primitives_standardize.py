#
#                                                                  gemini_python
#
#                                                      primitives_standardize.py
# ------------------------------------------------------------------------------
import os

import numpy as np
from importlib import import_module
from scipy import ndimage
from copy import deepcopy

from astrodata.provenance import add_provenance
from gempy.gemini import gemini_tools as gt
from gempy.gemini import irafcompat
from gempy.adlibrary.manipulate_ad import rebin_data
from geminidr.gemini.lookups import DQ_definitions as DQ
from geminidr import PrimitivesBASE
from recipe_system.utils.md5 import md5sum
from recipe_system.utils.decorators import parameter_override, capture_provenance
from recipe_system import __version__ as rs_version


from . import parameters_standardize

# ------------------------------------------------------------------------------
@parameter_override
@capture_provenance
class Standardize(PrimitivesBASE):
    """
    This is the class containing all of the primitives used to standardize an
    AstroData object.
    """
    tagset = None

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_standardize)

    def addDQ(self, adinputs=None, **params):
        """
        This primitive is used to add a DQ extension to the input AstroData
        object. The value of a pixel in the DQ extension will be the sum of the
        following: (0=good, 1=bad pixel (found in bad pixel mask), 2=pixel is
        in the non-linear regime, 4=pixel is saturated). This primitive will
        trim the BPM to match the input AstroData object(s).

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        static_bpm: str
            Name of bad pixel mask ("default" -> use default from look-up table)
            If set to None, no static_bpm will be added.
        user_bpm: str
            Name of the bad pixel mask created by the user from flats and
            darks.  It is an optional BPM that can be added to the static one.
        add_illum_mask: bool
            add illumination mask?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys["addDQ"]
        sfx = params["suffix"]

        # Getting all the filenames first prevents reopening the same file
        # for each science AD
        static_bpm_list = params['static_bpm']
        user_bpm_list = params['user_bpm']

        if static_bpm_list == "default":
            static_bpm_list = self.caldb.get_processed_bpm(adinputs)
            if static_bpm_list is not None:
                static_bpm_list = static_bpm_list.files
            else:
                static_bpm_list = [None] * len(adinputs)

        for ad, static, user in zip(*gt.make_lists(adinputs, static_bpm_list,
                                                   user_bpm_list, force_ad=True)):
            if ad.phu.get(timestamp_key):
                log.warning('No changes will be made to {}, since it has '
                            'already been processed by addDQ'.format(ad.filename))
                continue

            if static is None:
                # So it can be zipped with the AD
                final_static = [None] * len(ad)
            else:
                log.stdinfo("Using {} as static BPM\n".format(static.filename))
                if static.binning() != ad.binning():
                    static = rebin_data(deepcopy(static), xbin=ad.detector_x_bin(),
                                        ybin=ad.detector_y_bin())
                final_static = gt.clip_auxiliary_data(ad, aux=static,
                                                      aux_type='bpm',
                                                      return_dtype=DQ.datatype)

            if user is None:
                final_user = [None] * len(ad)
            else:
                log.stdinfo("Using {} as user BPM".format(user.filename))
                if user.binning() != ad.binning():
                    user = rebin_data(deepcopy(user), xbin=ad.detector_x_bin(),
                                      ybin=ad.detector_y_bin())
                final_user = gt.clip_auxiliary_data(ad, aux=user,
                                                    aux_type='bpm',
                                                    return_dtype=DQ.datatype)

            if static is None and user is None:
                log.stdinfo(f"No BPMs found for {ad.filename} and none supplied by the user.\n")

            for ext, static_ext, user_ext in zip(ad, final_static, final_user):
                if ext.mask is not None:
                    log.warning(f'A mask already exists in extension {ext.id}')
                    continue

                non_linear_level = ext.non_linear_level()
                saturation_level = ext.saturation_level()

                # Need to create the array first for 3D raw F2 data, with 2D BPM
                ext.mask = np.zeros_like(ext.data, dtype=DQ.datatype)
                if static_ext is not None:
                    ext.mask |= static_ext.data
                if user_ext is not None:
                    ext.mask |= user_ext.data

                if saturation_level:
                    log.fullinfo('Flagging saturated pixels in {} extension '
                                 '{} above level {:.2f}'.format(
                                     ad.filename, ext.id, saturation_level))
                    ext.mask |= np.where(ext.data >= saturation_level,
                                         DQ.saturated, 0).astype(DQ.datatype)

                if non_linear_level:
                    if saturation_level:
                        if saturation_level > non_linear_level:
                            log.fullinfo('Flagging non-linear pixels in {} '
                                         'extension {} above level {:.2f}'
                                         .format(ad.filename, ext.id,
                                                 non_linear_level))
                            ext.mask |= np.where((ext.data >= non_linear_level) &
                                                 (ext.data < saturation_level),
                                                 DQ.non_linear, 0).astype(DQ.datatype)
                            # Readout modes of IR detectors can result in
                            # saturated pixels having values below the
                            # saturation level. Flag those. Assume we have an
                            # IR detector here because both non-linear and
                            # saturation levels are defined and nonlin<sat
                            regions, nregions = ndimage.label(
                                                ext.data < non_linear_level)
                            # In all my tests, region 1 has been the majority
                            # of the image; however, I cannot guarantee that
                            # this is always the case and therefore we should
                            # check the size of each region
                            region_sizes = ndimage.labeled_comprehension(
                                ext.data, regions, np.arange(1, nregions+1),
                                len, int, 0)
                            # First, assume all regions are saturated, and
                            # remove any very large ones. This is much faster
                            # than progressively adding each region to DQ
                            hidden_saturation_array = np.where(regions > 0,
                                                    4, 0).astype(DQ.datatype)
                            for region in range(1, nregions+1):
                                # Limit of 10000 pixels for a hole is a bit arbitrary
                                if region_sizes[region-1] > 10000:
                                    hidden_saturation_array[regions==region] = 0
                            ext.mask |= hidden_saturation_array

                        elif saturation_level < non_linear_level:
                            log.warning(f'{ad.filename} extension {ext.id} '
                                        'has saturation level less than '
                                        'non-linear level')
                        else:
                            log.fullinfo('Saturation and non-linear levels '
                                         'are the same for {}:{}. Only '
                                         'flagging saturated pixels'
                                         .format(ad.filename, ext.id))
                    else:
                        log.fullinfo('Flagging non-linear pixels in {}:{} '
                                     'above level {:.2f}'
                                     .format(ad.filename, ext.id,
                                             non_linear_level))
                        ext.mask |= np.where(ext.data >= non_linear_level,
                                             DQ.non_linear, 0).astype(DQ.datatype)
            if static and static.filename:
                add_provenance(ad, static.filename, md5sum(static.path) or "", self.myself())
            if user and user.filename:
                add_provenance(ad, user.filename, md5sum(user.path) or "", self.myself())

        # Handle latency if reqested
        if params.get("latency", False):
            try:
                adinputs = self.addLatencyToDQ(adinputs, time=params["time"],
                                               non_linear=params["non_linear"])
            except AttributeError:
                log.warning("addLatencyToDQ() not defined in primitivesClass "
                            + self.__class__.__name__)

        # Add the illumination mask if requested
        if params['add_illum_mask']:
            adinputs = self.addIllumMaskToDQ(
                adinputs, **self._inherit_params(params, "addIllumMaskToDQ"))

        # Timestamp and update filenames
        for ad in adinputs:
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def addIllumMaskToDQ(self, adinputs=None, suffix=None, illum_mask=None):
        """
        Adds an illumination mask to each AD object

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        mask: str/None
            name of illumination mask mask (None -> use default)
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        # Getting all the filenames first prevents reopening the same file
        # for each science AD
        if illum_mask is None:
            illum_mask = [self._get_illum_mask_filename(ad) for ad in adinputs]

        for ad, illum in zip(*gt.make_lists(adinputs, illum_mask, force_ad=True)):
            if ad.phu.get(timestamp_key):
                log.warning('No changes will be made to {}, since it has '
                    'already been processed by addIllumMaskToDQ'.
                            format(ad.filename))
                continue

            if illum is None:
                # So it can be zipped with the AD
                final_illum = [None] * len(ad)
            else:
                log.fullinfo("Using {} as illumination mask".format(illum.filename))
                final_illum = gt.clip_auxiliary_data(ad, aux=illum, aux_type='bpm',
                                          return_dtype=DQ.datatype)

            for ext, illum_ext in zip(ad, final_illum):
                if illum_ext is not None:
                    # Ensure we're only adding the unilluminated bit
                    iext = np.where(illum_ext.data > 0, DQ.unilluminated,
                                    0).astype(DQ.datatype)
                    ext.mask = iext if ext.mask is None else ext.mask | iext

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)

        return adinputs

    def addVAR(self, adinputs=None, **params):
        """
        This primitive adds noise components to the VAR plane of each extension
        of each input AstroData object (creating the VAR plane if necessary).
        The calculations for these components are abstracted out to separate
        methods that operate on an individual AD object in-place.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        read_noise: bool (optional, default: False)
            add the read noise component?
        poisson_noise: bool (optional, default: False)
            add the Poisson noise component?
        """
        log = self.log
        log.debug(gt.log_message("primitive", "addVAR", "starting"))
        timestamp_key = self.timestamp_keys["addVAR"]
        read_noise = params['read_noise']
        poisson_noise = params['poisson_noise']
        suffix = params['suffix']

        if read_noise:
            if poisson_noise:
                log.stdinfo('Adding the read noise component and the Poisson '
                            'noise component of the variance')
            else:
                log.stdinfo('Adding the read noise component of the variance')
        else:
            if poisson_noise:
                log.stdinfo('Adding the Poisson noise component of the variance')
            else:
                log.warning('Cannot add a variance extension since no variance '
                            'component has been selected')
                return adinputs

        for ad in adinputs:
            if read_noise:
                self._addReadNoise(ad)
            if poisson_noise:
                self._addPoissonNoise(ad)
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

    def makeIRAFCompatible(self, adinputs=None):
        """
        Add keywords to make the pipeline-processed file compatible
        with the tasks in the Gemini IRAF package. For Hamamatsu data, also
        trim off the 48/binning rows and 1 column that IRAF trims off.
        """
        log = self.log
        log.debug(gt.log_message('primitive', self.myself(), 'starting'))
        timestamp_key = self.timestamp_keys[self.myself()]

        for ad in adinputs:
            irafcompat.pipeline2iraf(ad)
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)

        return adinputs

    def prepare(self, adinputs=None, **params):
        """
        Validate and standardize the datasets to ensure compatibility
        with the subsequent primitives.  The outputs, if written to
        disk will be given the suffix "_prepared".

        Currently, there are no input parameters associated with
        this primitive.

        Parameters
        ----------
        adinputs : None or list
            Input files that will be prepared. If `None`, it runs on the
            list of AstroData objects in the main stream.
        suffix: str
            Suffix to be added to output files (Default: "_prepared").
        """
        log = self.log
        log.debug(gt.log_message("primitive", "prepare", "starting"))

        filenames = [ad.filename for ad in adinputs]
        paths = [ad.path for ad in adinputs]

        timestamp_key = self.timestamp_keys["prepare"]
        sfx = params["suffix"]
        for primitive in ('validateData', 'standardizeStructure',
                          'standardizeHeaders', 'standardizeWCS'):
            passed_params = self._inherit_params(params, primitive)
            adinputs = getattr(self, primitive)(adinputs, **passed_params)

        for ad in adinputs:
            gt.mark_history(ad, self.myself(), timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)
        for ad, filename, path in zip(adinputs, filenames, paths):
            if path:
                add_provenance(ad, filename, md5sum(path) or "", self.myself())
        return adinputs

    def standardizeHeaders(self, adinputs=None, **params):
        """
        This primitive is used to standardize the headers of data. It adds
        the ORIGNAME, PROCSOFT, PROCSVER and PROCMODE keywords and then calls
        the standardizeObservatoryHeaders and standardizeInstrumentHeaders
        primitives.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        for ad in adinputs:
            if 'ORIGNAME' not in ad.phu:
                ad.phu.set('ORIGNAME', ad.orig_filename,
                           'Original filename prior to processing')

            # These are used by FitsStorage / GOA when handling reduced data.
            # Deliberately overwrite any existing values
            ad.phu.set('PROCSOFT', 'DRAGONS', 'Data Processing Software used')
            ad.phu.set('PROCSVER', rs_version, 'DRAGONS software version')
            ad.phu.set('PROCMODE', self.mode, 'Processing Mode [sq|ql|qa]')

        adinputs = self.standardizeObservatoryHeaders(adinputs,
                    **self._inherit_params(params, "standardizeObservatoryHeaders"))
        adinputs = self.standardizeInstrumentHeaders(adinputs,
                    **self._inherit_params(params, "standardizeInstrumentHeaders",
                                           pass_suffix=True))
        return adinputs

    def standardizeInstrumentHeaders(self, adinputs=None, **params):
        return adinputs

    def standardizeObservatoryHeaders(self, adinputs=None, **params):
        return adinputs

    def standardizeStructure(self, adinputs=None, **params):
        return adinputs

    def standardizeWCS(self, adinputs=None, **params):
        return adinputs

    def validateData(self, adinputs=None, suffix=None, require_wcs=True):
        """
        This is the data validation primitive. It checks that the instrument
        matches the primitivesClass and that there are the correct number
        of extensions.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        require_wcs: bool
            do all extensions have to have a defined WCS?
        """
        log = self.log
        timestamp_key = self.timestamp_keys[self.myself()]
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        missing_wcs_list = []

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning(f"No changes will be made to {ad.filename}, since"
                            " it has already been processed by validateData")
                continue

            # Check that the input is appropriate for this primitivesClass
            # Only the instrument is checked
            inst_name = ad.instrument(generic=True)
            if not inst_name in self.tagset:
                prim_class_name = self.__class__.__name__
                raise ValueError(f"Input file {ad.filename} is {inst_name} data"
                                 f" and not suitable for {prim_class_name} "
                                 "class")

            # Report if this is an image without square binned pixels
            if 'IMAGE' in ad.tags:
                xbin = ad.detector_x_bin()
                ybin = ad.detector_y_bin()
                if xbin != ybin:
                    log.warning(f"Image {ad.filename} is {xbin} x {ybin} "
                                "binned data")

            if self._has_valid_extensions(ad):
                log.fullinfo(f"The input file has been validated: {ad.filename}"
                             f" contains {len(ad)} extension(s)")
            else:
                raise ValueError(f"The {len(ad)} extension(s) in {ad.filename} "
                                 "does not match the number of extensions "
                                 f"expected in raw {inst_name} data.")

            if require_wcs:
                missing_wcs_list.extend([f"{ad.filename}:{ext.id}"
                                         for ext in ad if ext.wcs is None])

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)

        if missing_wcs_list:
            msg = "The following extensions did not produce a valid WCS:\n    "
            msg += '\n    '.join(extstr for extstr in missing_wcs_list)
            raise ValueError(msg+"\n")

        return adinputs

    @staticmethod
    def _has_valid_extensions(ad):
        """Check the AD has a valid number of extensions"""
        return len(ad) == 1

    def _addPoissonNoise(self, ad, dtype=np.float32):
        """
        This primitive calculates the variance due to Poisson noise for each
        science extension in the input AstroData list. A variance plane is
        added, if it doesn't exist, or else the variance is added to the
        existing plane if there is no header keyword indicating this operation
        has already been performed.

        This primitive should be invoked by calling addVAR(poisson_noise=True)
        """
        log = self.log

        log.fullinfo("Adding Poisson noise to {}".format(ad.filename))
        tags = ad.tags
        if 'BIAS' in tags:
            log.warning("It is not recommended to add Poisson noise "
                        "to the variance of a bias frame")
        elif ('GMOS' in tags and
              self.timestamp_keys["biasCorrect"] not in ad.phu and
              self.timestamp_keys["subtractOverscan"] not in ad.phu):
            log.warning("It is not recommended to add Poisson noise to the"
                        " variance of data that still contain a bias level")

        for ext in ad:
            if 'poisson' in ext.hdr.get('VARNOISE', '').lower():
                log.warning("Poisson noise already added for "
                            f"{ad.filename} extension {ext.id}")
                continue
            var_array = np.where(ext.data > 0, ext.data, 0).astype(dtype)
            if not ext.is_coadds_summed():
                var_array /= ext.coadds()
            if ext.is_in_adu():
                var_array /= gt.array_from_descriptor_value(ext, "gain")
            if ext.variance is None:
                ext.variance = np.full_like(ext.data, var_array, dtype=dtype)
            else:
                ext.variance += var_array
            varnoise = ext.hdr.get('VARNOISE')
            if varnoise is None:
                ext.hdr.set('VARNOISE', 'Poisson',
                            self.keyword_comments['VARNOISE'])
            else:
                ext.hdr['VARNOISE'] += ', Poisson'

    def _addReadNoise(self, ad, dtype=np.float32):
        """
        This primitive calculates the variance due to read noise for each
        science extension in the input AstroData list. A variance plane is
        added, if it doesn't exist, or else the variance is added to the
        existing plane if there is no header keyword indicating this operation
        has already been performed.

        This method should be invoked by calling addVAR(read_noise=True)

        If an extension is composed of data from multiple amplifiers, the read
        noise can be added provided there are the same number of Sections in
        the data_section() descriptor as there are values in read_noise(). The
        read noise will also be added to the overscan regions if the descriptor
        returns Sections. If the data are in ADU, then the gain() descriptor
        must also return a list of the same length.
        """
        log = self.log

        log.fullinfo("Adding read noise to {}".format(ad.filename))
        for ext in ad:
            extver = ext.hdr['EXTVER']
            if 'read' in ext.hdr.get('VARNOISE', '').lower():
                log.warning("Read noise already added for "
                            f"{ad.filename} extension {ext.id}")
                continue
            var_array = (gt.array_from_descriptor_value(ext, "read_noise") /
                         (gt.array_from_descriptor_value(ext, "gain")
                          if ext.is_in_adu() else 1.0)) ** 2

            if ext.variance is None:
                ext.variance = np.full_like(ext.data, var_array, dtype=dtype)
            else:
                ext.variance += var_array
            varnoise = ext.hdr.get('VARNOISE')
            if varnoise is None:
                ext.hdr.set('VARNOISE', 'read',
                            self.keyword_comments['VARNOISE'])
            else:
                ext.hdr['VARNOISE'] += ', read'

    def _get_illum_mask_filename(self, ad):
        """
        Gets the illumMask filename for an input science frame, using
        illumMask_dict in geminidr.<instrument>.lookups.maskdb.py and looks
        for a key <INSTRUMENT>_<MODE>_<XBIN><YBIN>. This file will be sent
        to clip_auxiliary_data for a subframe ROI.

        Returns
        -------
        str/None: Filename of the appropriate illumination mask
        """
        log = self.log
        inst = ad.instrument()
        mode = ad.tags & {'IMAGE', 'SPECT'}
        if mode:
            mode = mode.pop()
        else:  # DARK/BIAS (only F2 K-band darks for flats should get here)
            log.fullinfo("{} in neither IMAGE nor SPECT so does not require "
                         "an illumination mask".format(ad.filename))
            return None
        xbin = ad.detector_x_bin()
        ybin = ad.detector_y_bin()
        try:
            masks = import_module('.maskdb', self.inst_lookups)
            illum_dict = getattr(masks, 'illumMask_dict')
        except:
            log.fullinfo('No illumination mask dict for {}'.
                         format(ad.filename))
            return None

        # We've successfully loaded the illumMask_dict
        bpm_dir = os.path.join(os.path.dirname(masks.__file__), 'BPM')
        key = '{}_{}_{}{}'.format(inst, mode, xbin, ybin)
        try:
            mask = illum_dict[key]
        except KeyError:
            log.warning('No illumination mask found for {}'.format(ad.filename))
            return None
        # Prepend standard path if the filename doesn't start with '/'
        return mask if mask.startswith(os.path.sep) else \
            os.path.join(bpm_dir, mask)
