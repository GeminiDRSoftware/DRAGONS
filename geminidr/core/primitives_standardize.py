import astrodata
import gemini_instruments
from gempy.gemini import gemini_tools as gt
from gempy.gemini import irafcompat

from ..gemini.lookups import BPMDict
from ..gemini.lookups import MDFDict
from ..gemini.lookups import DQ_definitions as DQ

from .. import PrimitivesBASE
from .parameters_standardize import ParametersStandardize

from recipe_system.utils.decorators import parameter_override

import os
import shutil
import numpy as np
from scipy.ndimage import measurements
# ------------------------------------------------------------------------------
@parameter_override
class Standardize(PrimitivesBASE):
    """
    This is the class containing all of the primitives used to standardize an
    AstroData object.

    """
    tagset = None

    def __init__(self, adinputs, context, ucals=None, uparms=None):
        super(Standardize, self).__init__(adinputs, context, ucals=ucals, uparms=uparms)
        self.parameters = ParametersStandardize


    def addDQ(self, adinputs=None, stream='main', **params):
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
        sfx = self.parameters.addDQ["suffix"]
        dq_dtype = np.uint8

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning('No changes will be made to {}, since it has '
                            'already been processed by add DQ'.
                            format(ad.filename))
                continue

            bpm = self.parameters.addDQ['bpm']
            if bpm is None:
                bpm = self._get_bpm_filename(ad)
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

                ext.mask = bpm_ext.data
                if saturation_level:
                    log.fullinfo('Flagging saturated pixels in {} extver {} '
                                 'above level {:.2f}'.
                                 format(ad.filename, extver, saturation_level))
                    ext.mask |= np.where(ext.data >= saturation_level,
                                         DQ.saturated, 0)

                if non_linear_level:
                    if saturation_level:
                        if saturation_level > non_linear_level:
                            log.fullinfo('Flagging non-linear pixels in {} '
                                         'extver {} above level {:.2f}'.
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
                            log.warning('{} extver {} has saturation level '
                                        'less than non-linear level'.
                                        format(ad.filename, extver))
                        else:
                            log.fullinfo('Saturation and non-linear levels '
                                         'are the same for {} extver {}. Only '
                                         'flagging saturated pixels'.
                                format(ad.filename, extver))
                    else:
                        log.fullinfo('Flagging non-linear pixels in {} '
                                     'extver {} above level {:.2f}'.
                                     format(ad.filename, extver,
                                            non_linear_level))
                        ext.mask |= np.where(ext.data >= non_linear_level,
                                             DQ.non_linear, 0)

            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)

        # Add the illumination mask if requested
        if self.parameters.addDQ['illum_mask']:
            self.addIllumMaskToDQ(adinputs)

        return adinputs

    def addIllumMaskToDQ(self, adinputs=None, stream='main', **params):
        return adinputs

    def addMDF(self, adinputs=None, stream='main', **params):
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
        sfx = self.parameters.addMDF["suffix"]

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

            mdf = self.parameters.addMDF['mdf']
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
                    mdf = os.path.sep.join(MDFDict.__file__.split(os.path.sep)[:-3] +
                                            [inst.lower(), 'lookups', 'MDF',
                                            MDFDict.bpm_dict[key]])
                except KeyError:
                    # Look through the possible MDF locations
                    mdf = mask_name if mask_name.endswidth('.fits') else \
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

    def addVAR(self, adinputs=None, stream='main', **params):
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
        sfx = self.parameters.addVAR["suffix"]

        read_noise = self.parameters.addVAR['read_noise']
        poisson_noise = self.parameters.addVAR['poisson_noise']
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
                return

        for ad in adinputs:
            tags = ad.tags
            if poisson_noise and 'BIAS' in ad.tags:
                log.warning("It is not recommended to add a poisson noise "
                            "component to the variance of a bias frame")
            if (poisson_noise and 'GMOS' in ad.tags and
                ad.phu.get(self.timestamp_keys['subtractBias']) is not None):
                log.warning("It is not recommended to calculate a poisson "
                            "noise component of the variance using data that "
                            "still contains a bias level")

            self._calculate_var(ad, read_noise, poisson_noise)
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)

        return adinputs

    def makeIRAFCompatible(self, adinputs=None, stream='main', **params):
        """
        Add keywords to make the pipeline-processed file compatible
        with the tasks in the Gemini IRAF package.
        """
        log = self.log
        log.debug(gt.log_message('primitive', 'makeIRAFCompatible',
                                 'starting'))
        timestamp_key = self.timestamp_keys['makeIRAFCompatible']

        for ad in adinputs:
            irafcompat.pipeline2iraf(ad)
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)

        return adinputs

    def prepare(self, adinputs=None, stream='main', **params):
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
        sfx = self.parameters.prepare["suffix"]
        adinputs = self.validateData(adinputs)
        adinputs = self.standardizeStructure(adinputs)
        adinputs = self.standardizeObservatoryHeaders(adinputs)
        adinputs = self.standardizeInstrumentHeaders(adinputs)
        for ad in adinputs:
            gt.mark_history(ad, self.myself(), timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
        return adinputs

    def standardizeHeaders(self, adinputs=None, stream='main', **params):
        log = self.log
        log.debug(gt.log_message("primitive", "standardizeHeaders",
                                 "starting"))
        self.standardizeObservatoryHeaders(adinputs)
        self.standardizeInstrumentHeaders(adinputs)
        return adinputs

    def standardizeInstrumentHeaders(self, adinputs=None, stream='main',
                                     **params):
        return adinputs

    def standardizeObservatoryHeaders(self, adinputs=None, stream='main',
                                      **params):
        return adinputs

    def standardizeStructure(self, adinputs=None, stream='main', **params):
        return adinputs

    def validateData(self, adinputs=None, stream='main', **params):
        return adinputs

    ##########################################################################
    # Below are the helper functions for the primitives in this module       #
    ##########################################################################

    def _get_bpm_filename(self, ad):
        """
        Gets a bad pixel mask for an input science frame

        Parameters
        ----------
        adinput: AstroData
            AD instance for which we want a bpm

        Returns
        -------
        str:
            Filename of the appropriate bpm
        """
        inst = ad.instrument()
        xbin = ad.detector_x_bin()
        ybin = ad.detector_y_bin()
        if 'GMOS' in inst:
            det = ad.detector_name(pretty=True)[:3]
            amps = '{}amp'.format(3 * ad.phu.NAMPS)
            mos = '_mosaic' if (ad.phu.get(self.timestamp_keys['mosaicDetectors'])
                or ad.phu.get(self.timestamp_keys['tileArrays'])) else ''
            key = '{}_{}_{}{}_{}_{}{}'.format(inst, det, xbin, ybin, amps,
                                              'v1', mos)
            inst = 'GMOS'
        else:
            key = '{}_{}_{}'.format(inst, xbin, ybin)

        filename = os.path.sep.join(BPMDict.__file__.split(os.path.sep)[:-3] +
                                    [inst.lower(), 'lookups', 'BPM',
                                    BPMDict.bpm_dict[key]])
        return filename

    def _calculate_var(self, adinput, add_read_noise=False,
                       add_poisson_noise=False):
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
        log = self.log
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

            # Attach to the extension
            ext.variance = var_array.astype(var_dtype)

        return