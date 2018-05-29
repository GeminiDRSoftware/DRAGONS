#
#                                                                  gemini_python
#
#                                                      primitives_standardize.py
# ------------------------------------------------------------------------------
import os
import numpy as np
from importlib import import_module
from scipy.ndimage import measurements

from gempy.gemini import gemini_tools as gt
from gempy.gemini import irafcompat
from gempy.utils import logutils

from geminidr.gemini.lookups import DQ_definitions as DQ

from geminidr import PrimitivesBASE
from . import parameters_standardize

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class Standardize(PrimitivesBASE):
    """
    This is the class containing all of the primitives used to standardize an
    AstroData object.
    """
    tagset = None

    def __init__(self, adinputs, **kwargs):
        super(Standardize, self).__init__(adinputs, **kwargs)
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
        illum_mask: bool
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
            static_bpm_list = [self._get_bpm_filename(ad) for ad in adinputs]

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
                log.fullinfo("Using {} as static BPM".format(static.filename))
                final_static = gt.clip_auxiliary_data(ad, aux=static,
                                        aux_type='bpm', return_dtype=DQ.datatype)

            if user is None:
                final_user = [None] * len(ad)
            else:
                log.fullinfo("Using {} as user BPM".format(user.filename))
                final_user = gt.clip_auxiliary_data(ad, aux=user,
                                        aux_type='bpm', return_dtype=DQ.datatype)

            for ext, static_ext, user_ext in zip(ad, final_static, final_user):
                extver = ext.hdr['EXTVER']
                if ext.mask is not None:
                    log.warning('A mask already exists in extver {}'.
                                format(extver))
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
                    log.fullinfo('Flagging saturated pixels in {}:{} '
                                 'above level {:.2f}'.
                                 format(ad.filename, extver, saturation_level))
                    ext.mask |= np.where(ext.data >= saturation_level,
                                         DQ.saturated, 0).astype(DQ.datatype)

                if non_linear_level:
                    if saturation_level:
                        if saturation_level > non_linear_level:
                            log.fullinfo('Flagging non-linear pixels in {}:{} '
                                         'above level {:.2f}'.
                                         format(ad.filename, extver,
                                                non_linear_level))
                            ext.mask |= np.where((ext.data >= non_linear_level) &
                                                 (ext.data < saturation_level),
                                                 DQ.non_linear, 0).astype(DQ.datatype)
                            # Readout modes of IR detectors can result in
                            # saturated pixels having values below the
                            # saturation level. Flag those. Assume we have an
                            # IR detector here because both non-linear and
                            # saturation levels are defined and nonlin<sat
                            regions, nregions = measurements.label(
                                                ext.data < non_linear_level)
                            # In all my tests, region 1 has been the majority
                            # of the image; however, I cannot guarantee that
                            # this is always the case and therefore we should
                            # check the size of each region
                            region_sizes = measurements.labeled_comprehension(
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
                            log.warning('{}:{} has saturation level less than '
                                'non-linear level'.format(ad.filename, extver))
                        else:
                            log.fullinfo('Saturation and non-linear levels '
                                         'are the same for {}:{}. Only '
                                         'flagging saturated pixels'.
                                format(ad.filename, extver))
                    else:
                        log.fullinfo('Flagging non-linear pixels in {}:{} '
                                     'above level {:.2f}'.
                                     format(ad.filename, extver,
                                            non_linear_level))
                        ext.mask |= np.where(ext.data >= non_linear_level,
                                             DQ.non_linear, 0).astype(DQ.datatype)


        # Handle latency if reqested
        if params.get("latency", False):
            try:
                adinputs = self.addLatencyToDQ(adinputs)
            except AttributeError:
                log.warning("addLatencyToDQ() not defined in primitivesClass "
                            + self.__class__.__name__)

        # Add the illumination mask if requested
        if params['add_illum_mask']:
            adinputs = self.addIllumMaskToDQ(adinputs, illum_mask=params["illum_mask"])

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

    def addMDF(self, adinputs=None, suffix=None, mdf=None):
        """
        This primitive is used to add an MDF extension to the input AstroData
        object. If only one MDF is provided, that MDF will be add to all input
        AstroData object(s). If more than one MDF is provided, the number of
        MDF AstroData objects must match the number of input AstroData objects.
        If no MDF is provided, the primitive will attempt to determine an
        appropriate MDF.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        mdf: str/None
            name of MDF to add (None => use default)
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        if mdf is None:
            self.getMDF(adinputs)
            mdf_list = [self._get_cal(ad, 'mask') for ad in adinputs]
        else:
            mdf_list = mdf

        for ad, mdf in zip(*gt.make_lists(adinputs, mdf_list, force_ad=True)):
            if ad.phu.get(timestamp_key):
                log.warning('No changes will be made to {}, since it has '
                            'already been processed by addMDF'.
                            format(ad.filename))
                continue
            if hasattr(ad, 'MDF'):
                log.warning('An MDF extension already exists in {}, so no '
                            'MDF will be added'.format(ad.filename))
                continue

            if 'SPECT' not in ad.tags:
                log.stdinfo('{} is not spectroscopic data, so no MDF will '
                            'be added'.format(ad.filename))
                continue

            if mdf is None:
                log.stdinfo('No MDF could be retrieved for {}'.
                            format(ad.filename))
                continue

            try:
                # This will raise some sort of exception unless the MDF file
                # has a single MDF Table extension
                ad.MDF = mdf.MDF
            except:
                if len(mdf.tables) == 1:
                    ad.MDF = getattr(mdf, mdf.tables.pop())
                else:
                    log.warning('Cannot find MDF in {}, so no MDF will be '
                                'added'.format(mdf.filename))
                continue

            log.fullinfo('Attaching the MDF {} to {}'.format(mdf.filename,
                                                             ad.filename))

            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

    def addVAR(self, adinputs=None, **params):
        """
        This primitive calculates the variance of each science extension in the
        input AstroData object and adds the variance as an additional
        extension. This primitive will determine the units of the pixel data in
        the input science extension and calculate the variance in the same
        units. The two main components of the variance can be calculated and
        added separately, if desired, using the following formula:

        variance(read_noise) [electrons] = (read_noise [electrons])^2
        variance(read_noise) [ADU] = ((read_noise [electrons]) / gain)^2

        variance(poisson_noise) [electrons] =
            (number of electrons in that pixel)
        variance(poisson_noise) [ADU] =
            ((number of electrons in that pixel) / gain)

        The pixel data in the variance extensions will be the same size as the
        pixel data in the science extension.

        The read noise component of the variance can be calculated and added to
        the variance extension at any time, but should be done before
        performing operations with other datasets.

        The Poisson noise component of the variance can be calculated and added
        to the variance extension only after any bias levels have been
        subtracted from the pixel data in the science extension.

        The variance of a raw bias frame contains only a read noise component
        (which represents the uncertainty in the bias level of each pixel),
        since the Poisson noise component of a bias frame is meaningless.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        read_noise: bool
            add the read noise component?
        poisson_noise: bool
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
            tags = ad.tags
            if poisson_noise and 'BIAS' in tags:
                log.warning("It is not recommended to add a poisson noise "
                            "component to the variance of a bias frame")
            if (poisson_noise and 'GMOS' in tags and
                ad.phu.get(self.timestamp_keys['biasCorrect']) is None and
                ad.phu.get(self.timestamp_keys['subtractOverscan']) is None):
                log.warning("It is not recommended to calculate a poisson "
                            "noise component of the variance using data that "
                            "still contains a bias level")

            _calculate_var(ad, read_noise, poisson_noise)
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)

        return adinputs

    def makeIRAFCompatible(self, adinputs=None, stream='main', **params):
        """
        Add keywords to make the pipeline-processed file compatible
        with the tasks in the Gemini IRAF package.
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
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", "prepare", "starting"))
        timestamp_key = self.timestamp_keys["prepare"]
        sfx = params["suffix"]
        for primitive in ('validateData', 'standardizeStructure',
                          'standardizeHeaders'):
            passed_params = self._inherit_params(params, primitive)
            adinputs = getattr(self, primitive)(adinputs, **passed_params)

        for ad in adinputs:
            gt.mark_history(ad, self.myself(), timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)
        return adinputs

    def standardizeHeaders(self, adinputs=None, **params):
        """
        This primitive is used to standardize the headers of data. It adds
        the ORIGNAME keyword and then calls the standardizeObservatoryHeaders
        and standardizeInstrumentHeaders primitives.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        for ad in adinputs:
            if 'ORIGNAME' not in ad.phu:
                ad.phu.set('ORIGNAME', ad.orig_filename,
                           'Original filename prior to processing')

        adinputs = self.standardizeObservatoryHeaders(adinputs,
                    **self._inherit_params(params, "standardizeObservatoryHeaders"))
        adinputs = self.standardizeInstrumentHeaders(adinputs,
                    **self._inherit_params(params, "standardizeInstrumentHeaders",
                                           pass_suffix=True))
        return adinputs

    def standardizeInstrumentHeaders(self, adinputs=None, **params):
        return adinputs

    def standardizeObservatoryHeaders(self, adinputs=None, **params):
        """
        This primitive is used to make the changes and additions to the
        keywords in the headers of the data. All it does is add the ORIGNAME
        keyword to the header (if it doesn't exist), which is a standard
        DRAGONS keyword to assist with filename updates

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        for ad in adinputs:
            if timestamp_key in ad.phu:
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by standardize"
                            "ObservatoryHeaders".format(ad.filename))
                continue

            if 'ORIGNAME' not in ad.phu:
                ad.phu.set('ORIGNAME', ad.orig_filename,
                           'Original filename prior to processing')

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(ad, suffix=params["suffix"],
                                              strip=True)
        return adinputs

    def standardizeStructure(self, adinputs=None, **params):
        """
        This primitive is used to standardize the structure of GMOS data,
        specifically.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        attach_mdf: bool
            attach an MDF to the AD objects? (ignored if not tagged as SPECT)
        mdf: str
            full path of the MDF to attach
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        adoutputs = []
        # If attach_mdf=False, this just zips up the ADs with a list of Nones,
        # which has no side-effects.
        for ad, mdf in zip(*gt.make_lists(adinputs, params['mdf'])):
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by standardizeStructure".
                            format(ad.filename))
                adoutputs.append(ad)
                continue

            # Attach an MDF to each input AstroData object
            if params["attach_mdf"] and 'SPECT' in ad.tags:
                ad = self.addMDF([ad], mdf=mdf)[0]

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)
            adoutputs.append(ad)
        return adoutputs

    def validateData(self, adinputs=None, suffix=None):
        """
        This is the data validation primitive. It checks that the instrument
        matches the primitivesClass and that there are the correct number
        of extensions.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        timestamp_key = self.timestamp_keys[self.myself()]
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by validateData".
                            format(ad.filename))
                continue

            # Check that the input is appropriate for this primitivesClass
            # Only the instrument is checked
            inst_name = ad.instrument(generic=True)
            if not inst_name in self.tagset:
                prim_class_name = self.__class__.__name__
                raise IOError("Input file {} is {} data and not suitable for "
                    "{} class".format(ad.filename, inst_name, prim_class_name))

            # Report if this is an image without square binned pixels
            if 'IMAGE' in ad.tags:
                xbin = ad.detector_x_bin()
                ybin = ad.detector_y_bin()
                if xbin != ybin:
                    log.warning("Image {} is {} x {} binned data".
                                format(ad.filename, xbin, ybin))

            if self._has_valid_extensions(ad):
                log.fullinfo("The input file has been validated: {} contains "
                             "{} extension(s)".format(ad.filename, len(ad)))
            else:
                raise IOError("The {} extension(s) in {} does not match the "
                              "number of extensions expected in raw {} "
                              "data.".format(ad.filename, inst_name))

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

    @staticmethod
    def _has_valid_extensions(ad):
        """Check the AD has a valid number of extensions"""
        return len(ad) == 1

    def _get_bpm_filename(self, ad):
        """
        Gets the BPM filename for an input science frame. Takes bpm_dict from
        geminidr.<instrument>.lookups.maskdb.py and looks for a key
        <INSTRUMENT>_<XBIN><YBIN>. As a backup, uses the file in
        geminidr/<instrument>/lookups/BPM/ if there's only one file. This
        will be sent to clip_auxiliary_data for a subframe ROI.

        Returns
        -------
        str/None: Filename of the appropriate bpm
        """
        log = self.log
        inst = ad.instrument()
        xbin = ad.detector_x_bin()
        ybin = ad.detector_y_bin()
        bpm = None

        try:
            masks = import_module('.maskdb', self.inst_lookups)
            bpm_dir = os.path.join(os.path.dirname(masks.__file__), 'BPM')
            bpm_dict = getattr(masks, 'bpm_dict')
            key = '{}_{}{}'.format(inst, xbin, ybin)
            try:
                bpm = bpm_dict[key]
            except KeyError:
                log.warning('No static BPM found for {}'.format(ad.filename))
        except:
            log.warning('No static BPMs defined')

        if bpm is not None:
            # Prepend standard path if the filename doesn't start with '/'
            return bpm if bpm.startswith(os.path.sep) else \
                os.path.join(bpm_dir, bpm)
        return None

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
        mode = 'IMAGE' if 'IMAGE' in ad.tags else 'SPECT'
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

##########################################################################
# Below are the helper functions for the primitives in this module       #
##########################################################################

def _calculate_var(adinput, add_read_noise=False, add_poisson_noise=False):
    """
    Calculates the variance of each extension in the input AstroData
    object and updates the .variance attribute

    Parameters
    ----------
    adinput: AstroData
        AD instance to add variance planes to
    add_read_noise: bool
        add the read noise component?
    add_poisson_noise: bool
        add the Poisson noise component?
    """
    log = logutils.get_logger(__name__)
    gain_list = adinput.gain()
    read_noise_list = adinput.read_noise()
    var_dtype = np.float32

    for ext, gain, read_noise in zip(adinput, gain_list, read_noise_list):
        extver = ext.hdr['EXTVER']
        # Assume units are ADU if not explicitly given
        bunit = ext.hdr.get('BUNIT', 'ADU')

        # Create a variance array with the read noise (or zero)
        if add_read_noise:
            if read_noise is None:
                log.warning('Read noise for {} extver {} = None. Setting '
                            'to zero'.format(adinput.filename, extver))
                read_noise = 0.0
            else:
                log.fullinfo('Read noise for {} extver {} = {} electrons'.
                         format(adinput.filename, extver, read_noise))
                log.fullinfo('Calculating the read noise component of '
                             'the variance in {}'.format(bunit))
                if bunit.upper() == 'ADU':
                    read_noise /= gain
            var_array = np.full(ext.data.shape, read_noise*read_noise)
        else:
            var_array = np.zeros(ext.data.shape)

        # Add the Poisson noise if desired
        if add_poisson_noise:
            poisson_array = (ext.data if ext.is_coadds_summed() else
                             ext.data / ext.coadds())
            if bunit.upper() == 'ADU':
                poisson_array = poisson_array / gain
            log.fullinfo('Calculating the Poisson noise component of '
                         'the variance in {}'.format(bunit))
            var_array += np.where(poisson_array > 0, poisson_array, 0)

        if ext.variance is not None:
            if add_read_noise and add_poisson_noise:
                raise ValueError("Cannot add read noise and Poisson noise"
                                 " components to variance as variance "
                                 "already exists")
            else:
                log.fullinfo("Combining the newly calculated variance "
                             "with the current variance extension {}:{}".
                             format(ext.filename, extver))
                var_array += ext.variance
        else:
            log.fullinfo("Adding variance to {}:{}".format(ext.filename,
                                                           extver))
        # Attach to the extension
        ext.variance = var_array.astype(var_dtype)
    return
