import os
import shutil
import numpy as np
from importlib import import_module

from scipy.ndimage import measurements

import astrodata
import gemini_instruments

from gempy.gemini import gemini_tools as gt
from gempy.gemini import irafcompat

from geminidr.gemini.lookups import MDFDict
from geminidr.gemini.lookups import DQ_definitions as DQ

from geminidr import PrimitivesBASE
from geminidr.core.parameters_standardize import ParametersStandardize
from gempy.utils import logutils

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
        self.parameters = ParametersStandardize

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
        bpm: str/None
            name of bad pixel mask (None -> use default)
        illum_mask: bool
            add illumination mask?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys["addDQ"]
        sfx = params["suffix"]
        dq_dtype = np.int16

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning('No changes will be made to {}, since it has '
                    'already been processed by add DQ'.format(ad.filename))
                continue

            bpm = params['bpm']
            if bpm is None:
                bpm = self._get_bpm_filename(ad)

            if bpm is None:
                final_bpm = [None] * len(ad)
            else:
                log.fullinfo("Using {} as BPM".format(bpm))
                bpm_ad = astrodata.open(bpm)

                clip_method = gt.clip_auxiliary_data_GSAOI if 'GSAOI' in ad.tags \
                    else gt.clip_auxiliary_data
                final_bpm = clip_method(ad, bpm_ad, 'bpm', dq_dtype,
                                        self.keyword_comments)

            for ext, bpm_ext in zip(ad, final_bpm):
                extver = ext.hdr.EXTVER
                if ext.mask is not None:
                    log.warning('A mask already exists in extver {}'.
                                format(extver))
                    continue

                non_linear_level = ext.non_linear_level()
                saturation_level = ext.saturation_level()

                ext.mask = bpm_ext.data if bpm_ext is not None else \
                    np.zeros_like(ext.data, dtype=np.int16)
                if saturation_level:
                    log.fullinfo('Flagging saturated pixels in {}:{} '
                                 'above level {:.2f}'.
                                 format(ad.filename, extver, saturation_level))
                    ext.mask |= np.where(ext.data >= saturation_level,
                                         DQ.saturated, 0)

                if non_linear_level:
                    if saturation_level:
                        if saturation_level > non_linear_level:
                            log.fullinfo('Flagging non-linear pixels in {}:{} '
                                         'above level {:.2f}'.
                                         format(ad.filename, extver,
                                                non_linear_level))
                            ext.mask |= np.where((ext.data >= non_linear_level) &
                                                 (ext.data < saturation_level),
                                                 DQ.non_linear, 0)
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
                                                            4, 0).astype(dq_dtype)
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
                                             DQ.non_linear, 0)

            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)

        # Add the illumination mask if requested
        if params['illum_mask']:
            self.addIllumMaskToDQ(adinputs)

        return adinputs

    def addIllumMaskToDQ(self, adinputs=None, **params):
        return adinputs

    def addMDF(self, adinputs=None, **params):
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
        log.debug(gt.log_message("primitive", "addMDF", "starting"))
        timestamp_key = self.timestamp_keys["addMDF"]
        sfx = params["suffix"]

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning('No changes will be made to {}, since it has '
                            'already been processed by addMDF'.
                            format(ad.filename))
                continue

            if 'SPECT' not in ad.tags:
                log.stdinfo('{} is not spectroscopic data, so no MDF will '
                            'be added'.format(ad.filename))
                continue

            if hasattr(ad, 'MDF'):
                log.warning('An MDF extension already exists in {}, so no '
                            'MDF will be added'.format(ad.filename))
                continue

            mdf = params['mdf']
            if mdf is None:
                try:
                    inst = ad.instrument()
                    mask_name = ad.phu.MASKNAME
                    key = '{}_{}'.format(inst, mask_name)
                except AttributeError:
                    log.warning('Unable to create the key for the lookup '
                                'table so no MDF will be added')
                    continue

                try:
                    # Look in the instrument MDF directory
                    mdf = os.path.join(self.dr_root, inst.lower(), 'lookups',
                                       'MDF', MDFDict.mdf_dict[key])
                except KeyError:
                    # Look through the possible MDF locations
                    mdf = mask_name if mask_name.endswith('.fits') else \
                        '{}.fits'.format(mask_name)
                    for location in MDFDict.mdf_locations:
                        fullname = os.path.join(os.path.sep, location, mdf)
                        if os.path.exists(fullname):
                            # Copy MDF to local directory if it's elsewhere
                            if location != '.':
                                shutil.copy(fullname, '.')
                            break
                    else:
                        log.warning('The MDF {} was not found in any of the '
                                    'search directories, so no MDF will be '
                                    'added'.format(mdf))
                        continue

            try:
                # This will raise some sort of exception unless the MDF file
                # has a single MDF Table extension
                ad.MDF = astrodata.open(mdf).MDF
            except:
                log.warning('Cannot convert {} to AstroData object, so no '
                            'MDF will be added'.format(mdf))
                continue

            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
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
                ad.phu.get(self.timestamp_keys['subtractBias']) is None):
                log.warning("It is not recommended to calculate a poisson "
                            "noise component of the variance using data that "
                            "still contains a bias level")

            _calculate_var(ad, read_noise, poisson_noise)
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=suffix, strip=True)

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
        adinputs = self.validateData(adinputs)
        adinputs = self.standardizeStructure(adinputs)
        adinputs = self.standardizeObservatoryHeaders(adinputs)
        adinputs = self.standardizeInstrumentHeaders(adinputs)
        for ad in adinputs:
            gt.mark_history(ad, self.myself(), timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
        return adinputs

    def standardizeHeaders(self, adinputs=None, **params):
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        adinputs = self.standardizeObservatoryHeaders(adinputs, **params)
        adinputs = self.standardizeInstrumentHeaders(adinputs, **params)
        return adinputs

    def standardizeInstrumentHeaders(self, adinputs=None, **params):
        return adinputs

    def standardizeObservatoryHeaders(self, adinputs=None, **params):
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
            ad.filename = gt.filename_updater(adinput=ad, suffix=params["suffix"],
                                              strip=True)
            adoutputs.append(ad)
        return adoutputs

    def validateData(self, adinputs=None, **params):
        """
        This is the generic data validation primitive, for data which do not
        require any specific validation checks. It timestamps and moves on.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        repair: bool
            Repair the data, if necessary? This does not work yet!
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by validateData".
                            format(ad.filename))
                continue

            # Report if this is an image without square binned pixels
            if 'IMAGE' in ad.tags:
                xbin = ad.detector_x_bin()
                ybin = ad.detector_y_bin()
                if xbin != ybin:
                    log.warning("Image {} is {} x {} binned data".
                                format(ad.filename, xbin, ybin))

            try:
                valid_num_ext = params['num_exts']
            except KeyError:
                log.status("No validation required for {}".format(ad.filename))
            else:
                if not isinstance(valid_num_ext, list):
                    valid_num_ext = [valid_num_ext]
                num_ext = len(ad)
                if num_ext in valid_num_ext:
                    log.fullinfo("The input file has been validated: {} "
                             "contains {} extension(s)".format(ad.filename,
                                                               num_ext))
                else:
                    if params['repair']:
                        # Something could be done here
                        pass
                    raise IOError("The number of extensions in {} does not "
                                "match the number of extensions expected "
                                "in raw {} data.".format(ad.filename,
                                                         ad.instrument()))

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=params["suffix"],
                                              strip=True)
        return adinputs

##########################################################################
# Below are the helper functions for the primitives in this module       #
##########################################################################

    def _get_bpm_filename(self, ad):
        """
        Gets a bad pixel mask for an input science frame. Takes bpm_dict from
        geminidr.<instrument>.lookups.mask_dict and looks for a key
        <INSTRUMENT>_<XBIN>_<YBIN>. As a backup, uses the dict value if there's
        only one entry, or the file in geminidr/<instrument>/lookups/BPM/ if
        there's only one file.

        Parameters
        ----------
        adinput: AstroData
            AD instance for which we want a bpm

        Returns
        -------
        str: Filename of the appropriate bpm
        """
        log = self.log
        inst = ad.instrument()
        xbin = ad.detector_x_bin()
        ybin = ad.detector_y_bin()
        key = '{}_{}_{}'.format(inst, xbin, ybin)
        bpm_dir = os.path.join(self.dr_root, inst.lower(), 'lookups', 'BPM')
        bpm_pkg = 'geminidr.{}.lookups'.format(inst.lower())

        try:
            masks = import_module('.mask_dict', bpm_pkg)
        except ImportError:
            pass
        else:
            bpm_dict = masks.bpm_dict
            try:
                return os.path.join(bpm_dir, bpm_dict[key])
            except KeyError:
                if len(bpm_dict) == 1:
                    bpm = bpm_dict.values()[0]
                    log.stdinfo('Only one entry in BPM dict. Using {} as BPM'.
                                format(bpm))
                    return bpm

        # No help from the dict.
        # Look in the BPM directory; return a file if there's only one
        try:
            bpm_files = [file for file in os.listdir(bpm_dir) if
                         file.endswith('.fits')]
        except OSError:
            # Directory doesn't exist
            pass
        else:
            if len(bpm_files) == 1:
                bpm = bpm_files[0]
                log.stdinfo('Found single image in BPM directory. Using {} as BPM'.
                            format(bpm))
                return bpm
        log.stdinfo('No BPM found for {}'.format(ad.filename))
        return None

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

    Returns
    -------
    AstroData:
        an updated AD instance
    """
    log = logutils.get_logger(__name__)
    gain_list = adinput.gain()
    read_noise_list = adinput.read_noise()
    var_dtype = np.float32

    for ext, gain, read_noise in zip(adinput, gain_list, read_noise_list):
        extver = ext.hdr.EXTVER
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
                poisson_array /= gain
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