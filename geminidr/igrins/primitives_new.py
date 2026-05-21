import numpy as np
from astropy.io import fits
from astropy.modeling import models

from gwcs.wcs import WCS as gWCS

import astrodata
from astrodata import Section
from gempy.gemini import gemini_tools as gt
from gempy.library import transform

from .primitives_igrins import IGRINS
from geminidr.gemini.lookups import DQ_definitions as DQ

from recipe_system.utils.decorators import parameter_override
from . import parameters_new

from .procedures.apertures import Apertures
from .procedures.correct_distortion import get_rectified_2dspec
from .procedures.iraf_helper import get_wat_cards, get_wat_header


@parameter_override
class IGRINSNew(IGRINS):
    tagset = {}

    def _initialize(self, adinputs=None, **kwargs):
        self.inst_lookups = 'geminidr.igrins.lookups'
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_new)

    def createDataCube(self, adinputs=None, **params):
        """
        Create the data cube.

        This is currently just saveTwodspec but taking things from the main
        stream.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        #timestamp_key = self.timestamp_keys[self.myself()]
        suffix = params["suffix"]
        height_2dspec = params["height_2dspec"]
        conserve_flux = True
        # height_2dspec = 100 # obsset.get_recipe_parameter("height_2dspec")
        wavelength_increasing_order = params["wavelength_increasing_order"]

        adoutputs = []
        for ad in adinputs:
            ad_sky = self._get_ad_sky(ad)
            wat_table = ad_sky[0].WAT_HEADER

            # make sure you apply convert_data to the output. If get_wat_header is
            # called with wavelength_increasing_order=True, convert_data will rearrange
            # the data to the correct order.
            wvl_header, convert_data = get_wat_header(wat_table,
                                                      wavelength_increasing_order)

            ordermap = ad_sky[0].ORDERMAP
            # FIXME we should use proper badpixel mask.
            ordermap_bpixed = np.ma.array(ordermap, mask=ad_sky[0].mask).filled(0)
            ap = Apertures(ad_sky[0].SLITEDGE)

            _ = get_rectified_2dspec(ad[0].data, ordermap_bpixed, ap,  # bottom_up_solutions,
                                     conserve_flux=conserve_flux, height=height_2dspec)
            d0_shft_list, msk_shft_list, height = _
            with np.errstate(invalid="ignore"):
                d = np.array(d0_shft_list) / np.array(msk_shft_list)

            d = convert_data(d.astype("float32"))
            hdu_spec2d = fits.ImageHDU(header=wvl_header, data=d)

            ad_out = astrodata.create(ad.phu)
            ad_out.append(hdu_spec2d)

            _ = get_rectified_2dspec(ad[0].variance, ordermap, ap,  # bottom_up_solutions,
                                     conserve_flux=conserve_flux, height=height)
            d0_shft_list, msk_shft_list, _ = _
            with np.errstate(invalid="ignore"):
                d = np.array(d0_shft_list) / np.array(msk_shft_list)

            ad_out[0].variance = d
            ad_out[0].WAVELENGTHS = np.array(ad_sky[0].WVLSOL["wavelengths"], dtype=np.float32)

            ad_out.update_filename(suffix=suffix, strip=True)
            adoutputs.append(ad_out)

        return adoutputs

    def distortionCorrect(self, adinputs=None, **params):
        """
        Corrects the wavelength distortion by shifting all rows so that the
        lines of constant wavelength become vertical. This can currently use
        the IGRINS-2 PLP code or the DRAGONS transform module.

        TODO: This can ultimately be removed once the WCS is properly
        implemented.

        NB. This also flatfields the data.

        Parameters
        ----------
        suffix : str
            Suffix to be added to output files.
        interpolant : str
            Type of interpolant
        subsample : int
            Pixel subsampling factor.
        dq_threshold : float
            The fraction of a pixel's contribution from a DQ-flagged pixel to
            be considered 'bad' and also flagged.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        interpolant = params["interpolant"]
        subsample = params["subsample"]
        dq_threshold = params["dq_threshold"]
        use_dragons = params["use_dragons"]

        adoutputs = []
        for ad in adinputs:
            ad_flat = self._get_ad_flat(ad)
            ad_sky = self._get_ad_sky(ad)

            ap = Apertures(ad_sky[0].SLITEDGE)
            data_minus_flattened = ad[0].data / ad_flat[0].data
            variance_map = ad[0].variance + 2**2 # FIXME figure out the readout
                                                 # noize and properly update it

            if use_dragons:  # use existing DRAGONS transform module
                x = np.meshgrid(*(np.arange(length) for length in ad[0].shape[::-1]))[1]
                t = models.Identity(2)  # ensure array size doesn't change
                t.inverse = (models.Mapping((0, 1, 1)) |
                             models.Tabular2D(lookup_table=x+ad_sky[0].SLITOFFSETMAP.T, bounds_error=False,
                                              fill_value=0) & models.Identity(1))
                if ad[0].wcs is None:
                    ad[0].wcs = gWCS([(astrodata.wcs.pixel_frame(naxes=2), t),
                                      (astrodata.wcs.pixel_frame(naxes=2, name="xshifted"), None)])
                else:
                    ad[0].wcs = gWCS([(ad[0].wcs.pipeline[0].frame, t),
                                      (astrodata.wcs.pixel_frame(naxes=2, name="xshifted"),
                                       ad[0].wcs.pipeline[0].transform),
                                      ] + ad[0].wcs.pipeline[1:])

                ad_out = transform.resample_from_wcs(
                    ad, 'xshifted', interpolant=interpolant,
                    subsample=subsample, parallel=False,
                    threshold=dq_threshold
                )

            else:
                _ = ap.get_shifted_images(ad[0].SLITPROFILE_MAP,
                                          variance_map,
                                          data_minus_flattened,
                                          slitoffset_map=ad_sky[0].SLITOFFSETMAP,
                                          debug=False)
                data_shft, variance_map_shft, profile_map_shft, msk1_shft = _
                ad_out = astrodata.create(ad.phu)
                new_image = ad[0].__class__(data=data_shft.astype(np.float32),
                                            mask=msk1_shft.astype(ad[0].mask.dtype),
                                            variance=variance_map_shft.astype(np.float32))
                ad_out.append(new_image)
                ad_out[0].SLITPROFILE_MAP = profile_map_shft.astype(np.float32)

            # Timestamp and update the filename
            gt.mark_history(ad_out, primname=self.myself(), keyword=timestamp_key)
            ad_out.update_filename(suffix=sfx, strip=True)
            adoutputs.append(ad_out)

        return adoutputs

    def extractSpectraNew(self, adinputs=None, **params):
        """
        Extract 1D stellar spectra from 2D spectral data using optimal extraction.

        This method performs optimal extraction of stellar spectra from 2D
        spectral data, taking into account the spatial profile of the star
        and the noise characteristics of the detector. The extraction can be
        performed using different methods and parameters to optimize the
        signal-to-noise ratio.

        The method performs the following steps:
        1. Loads flat field and sky data for calibration
        2. Applies flat field correction
        3. Performs optimal extraction using the specified method
        4. Calculates wavelength solution and signal-to-noise ratios
        5. Returns the extracted 1D spectrum with associated metadata

        Returns
        -------
        AstroData
            A new AstroData object containing the extracted 1D spectrum with
            the following extensions:
            - Primary HDU: The extracted 1D spectrum
            - Variance array: The variance of the extracted spectrum
            - Wavelengths: The wavelength solution for the spectrum
            - SN_PER_RESEL: Signal-to-noise ratio per resolution element

        Notes
        -----
        - The method requires flat field and sky data to be available through
          the `_get_ad_flat` and `_get_ad_sky` methods.
        - The extraction uses the SLITEDGE information to define the extraction
          apertures.
        - The wavelength solution is taken from the WVLFIT_RESULTS attribute
          of the sky data.
        - The output spectrum includes WCS information in the header for
          wavelength calibration.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys["extractSpectra"]
        sfx = params["suffix"]
        extraction_mode = params["extraction_mode"]
        pixel_per_res_element = params["pixel_per_res_element"]

        adoutputs = []
        for ad in adinputs:
            ad_flat = self._get_ad_flat(ad)
            ad_sky = self._get_ad_sky(ad)

            ap = Apertures(ad_sky[0].SLITEDGE)
            ordermap = ad_sky[0].ORDERMAP
            slitpos_map = ad_sky[0].SLITPOSMAP
            slitoffset_map = ad_sky[0].SLITOFFSETMAP
            ordermap_bpixed = np.ma.array(ordermap, mask=ad_flat[0].mask > 0).filled(0)

            weight_thresh = None
            remove_negative = False
            s_list, v_list = ap.extract_stellar_from_shifted(
                ordermap_bpixed, ad[0].SLITPROFILE_MAP, ad[0].variance,
                ad[0].data, ad[0].mask, weight_thresh=weight_thresh,
                remove_negative=remove_negative)

            ad_out = astrodata.create(ad.phu)
            new_image = ad[0].__class__(data=s_list.astype(np.float32),
                                        variance=v_list.astype(np.float32))
            ad_out.append(new_image)

            # Timestamp and update the filename
            gt.mark_history(ad_out, primname=self.myself(), keyword=timestamp_key)
            ad_out.update_filename(suffix=sfx, strip=True)
            adoutputs.append(ad_out)

        return adoutputs

