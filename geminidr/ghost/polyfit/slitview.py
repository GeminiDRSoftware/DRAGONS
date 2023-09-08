import math
import numpy as np
from skimage import draw, transform, util
from scipy.optimize import least_squares
from astropy.modeling import models, fitting, Parameter, Fittable1DModel

# pylint: disable=maybe-no-member, too-many-instance-attributes

'''
# Rotation of slit on camera
ROTANGLE = 90.0-9.04

# Point around which to rotate
ROT_CENTER = [900.0, 780.0]

SLITVIEW_PARAMETERS = {
    'std': {
        'central_pix': {
            'red': [781, 985],
            'blue': [771, 946]
            #'red': [77, 65],    #
            #'blue': [77, 156]   #
        },
        'extract_half_width': 3,
        'sky_pix_only_boundaries': {
            'red': [47, 63],
            'blue': [47, 63]
        },
        'object_boundaries': {
            'red': [[3, 46], [64, 107]],
            'blue': [[3, 46], [64, 107]]
        },
        'sky_pix_boundaries': {
            'red': [3, 107],
            'blue': [3, 107]
        },
    },
    'high': {
        'central_pix': {
            'red': [770, 857],
            'blue': [760, 819],
            #'red': [78, 95], #
            #'blue': [78, 4]  #
        },
        'extract_half_width': 2,
        'sky_pix_only_boundaries': {
            'red': [82, 106],
            'blue': [82, 106]
        },
        'object_boundaries': {
            'red': [[11, 81], [4, 9]],
            'blue': [[11, 81], [4, 9]]
        },
        'sky_pix_boundaries': {
            'red': [11, 106], 'blue': [11, 106]},
    }
}
'''

S3 = np.sqrt(3)

# A quick counterpoint to the existing floordiv (which is just a // b)
def ceildiv(a, b):
    return -(a // -b)

class SlitView(object):
    """
    A class containing tools common to processing the dark and bias corrected
    slit-viewer images.

    Parameters
    ----------
    slit_image: :obj:`numpy.ndarray`
        A single slit viewer image, which has been processed, cosmic-ray
        rejection etc.

    flat_image: :obj:`numpy.ndarray`
        A single slit viewer flat field image, which has been processed,
        cosmic-ray rejection etc.

    microns_pix: float (optional)
        Scale in microns in the slit plane for each pixel in the slit view-
        ing camera. The default value assumes 2x2 binning of the slit view-
        ing camera. Default is ``4.54*180/50*2``.

    mode: string
        ``'std'`` or ``'high'``: the observing mode. Default is ``'std'``.

    slit_length: float (optional)
        Physical slit length to be extracted in microns. Default is ``3600.``.

    reverse_profile: bool
        Do we reverse the profile? This is a sign convention issue between
        the slit viewer and the CCD, to be determined through testing on
        real data.

    stowed: list
        are any IFUs stowed? this is relevant for constructing the object
        slit profiles as it cannot be discerned from the slit_image
    """
    def __init__(self, slit_image, flat_image, slitvpars, microns_pix=4.54*180/50,
                 binning=2, mode='std', slit_length=3600., stowed=None):
        self.binning = binning
        self.rota = slitvpars['rota']
        self.center = [slitvpars['rotyc'] // binning, slitvpars['rotxc'] // binning]
        self.central_pix = {
            'red': [slitvpars['center_y_red'] // binning,
                    slitvpars['center_x_red'] // binning],
            'blue': [slitvpars['center_y_blue'] // binning,
                     slitvpars['center_x_blue'] // binning]
        }
        self.sky_pix_only_boundaries = {
            'red': [slitvpars['skypix0'] // binning,
                    ceildiv(slitvpars['skypix1'], binning)],
            'blue': [slitvpars['skypix0'] // binning,
                     ceildiv(slitvpars['skypix1'], binning)]
        }
        self.object_boundaries = {
            'red': [[slitvpars['obj0pix0'] // binning,
                     ceildiv(slitvpars['obj0pix1'], binning)],
                    [slitvpars['obj1pix0'] // binning,
                     ceildiv(slitvpars['obj1pix1'], binning)]],
            'blue': [[slitvpars['obj0pix0'] // binning,
                     ceildiv(slitvpars['obj0pix1'], binning)],
                     [slitvpars['obj1pix0'] // binning,
                      ceildiv(slitvpars['obj1pix1'], binning)]],
        }
        if mode == 'std':
            self.sky_pix_boundaries = {
                'red': [slitvpars['obj0pix0'] // binning,
                        ceildiv(slitvpars['obj1pix1'], binning)],
                'blue': [slitvpars['obj0pix0'] // binning,
                        ceildiv(slitvpars['obj1pix1'], binning)]
            }
        else:
            self.sky_pix_boundaries = {
                'red': [slitvpars['obj0pix0'] // binning,
                        ceildiv(slitvpars['skypix1'], binning)],
                'blue': [slitvpars['obj0pix0'] // binning,
                        ceildiv(slitvpars['skypix1'], binning)]
            }

        self.mode = mode
        self.slit_length = slit_length
        self.microns_pix = microns_pix * binning
        self.reverse_profile = {'red': False, 'blue': True}

        self.extract_half_width = ceildiv(slitvpars['ext_hw'], binning)
        if slit_image is None or self.rota == 0.0:
            self.slit_image = slit_image
        else:
            self.slit_image = transform.rotate(util.img_as_float64(slit_image), self.rota, center=self.center)
        if flat_image is None or self.rota == 0.0:
            self.flat_image = flat_image
        else:
            self.flat_image = transform.rotate(util.img_as_float64(flat_image), self.rota, center=self.center)

        self.stowed = stowed or []

        # WARNING: These parameters below should be input from somewhere!!!
        # The central pixel in the y-direction (along-slit) defines the slit
        # profile offset, i.e. it interacts directly with the tramline fitting
        # and a change to one is a change to the other.
        # Co-ordinates are in standard python co-ordinates, i.e. y then x
        '''
        if mode in SLITVIEW_PARAMETERS.keys():
            for attr, value in SLITVIEW_PARAMETERS[mode].items():
                setattr(self, attr, value)
        else:
            raise ValueError("Invalid Mode: " + str(mode))
        '''

    def cutout(self, arm='red', use_flat=False):
        """
        Extract the 2-dimensional slit profile cutout.

        Parameters
        ----------
        arm: string, optional
            Either ``'red'`` or ``'blue'`` for GHOST. Default is ``'red'``.

        use_flat: bool, optional
            Cutout from the flat (True) or the slit frame (False). Default is
            False.

        Returns
        -------
        profile: :obj:`numpy.ndarray` (npix)
            The 2-dimensional slit profile cutout.
        """
        try:
            central_pix = self.central_pix[arm]
        except:
            raise ValueError("Invalid arm: '%s'" % arm)

        if use_flat:
            this_slit_image = self.flat_image
        else:
            this_slit_image = self.slit_image

        y_halfwidth = int(self.slit_length/self.microns_pix/2)
        return this_slit_image[
            central_pix[0]-y_halfwidth:central_pix[0]+y_halfwidth+1,
            central_pix[1]-self.extract_half_width:central_pix[1] +
            self.extract_half_width+1]

    def slit_profile(self, arm='red', return_centroid=False, use_flat=False,
                     denom_clamp=10, reverse_profile=None):
        """
        Extract the 1-dimensional slit profile.

        Parameters
        ----------
        arm: string, optional
            Either ``'red'`` or ``'blue'`` for GHOST. Default is ``'red'``.

        return_centroid: bool, optional
            Do we also return the pixel centroid of the slit? Default is False.

        use_flat: bool, optional
            Do we use the flat image? False for the object image.
            Default is False.

        denom_clamp: float, optional
            Denominator clamp - fluxes below this value are not used when
            computing the centroid. Defaults to ``10``.

        Returns
        -------
        profile: :obj:`numpy.ndarray` (npix)
            The summed 1-dimensional slit profile.
        """
        y_halfwidth = int(self.slit_length/self.microns_pix/2)
        cutout = self.cutout(arm, use_flat)
        if reverse_profile is None:
            reverse_profile = self.reverse_profile[arm]

        # Sum over the 2nd axis, i.e. the x-coordinate.
        profile = np.sum(cutout, axis=1)
        if reverse_profile:
            profile = profile[::-1]

        if return_centroid:
            xcoord = np.arange(
                -self.extract_half_width, self.extract_half_width+1)
            xcoord = np.tile(xcoord, 2*y_halfwidth+1).reshape(
                (2*y_halfwidth+1, 2*self.extract_half_width+1))
            centroid = np.sum(xcoord*cutout, 1)/np.maximum(profile, denom_clamp)
            return profile, centroid
        else:
            return profile

    def object_slit_profiles(self, arm='red', correct_for_sky=True,
                             used_objects=[0, 1],
                             append_sky=True, normalise_profiles=True):
        """
        Extract object slit profiles.

        Parameters
        ----------
        arm: string, optional
            Either ``'red'`` or ``'blue'`` for GHOST. Default is ``'red'``.

        correct_for_sky : bool, optional
            Should the slit profiles be corrected for sky? Defaults to True.

        append_sky : bool, optional
            Append the sky profile to the output ``profiles``? Defaults to True.

        normalise_profiles : bool, optional
            Should profiles be normalised? Defaults to True.

        used_objects: list of int, indices of used objects
            Denotes which objects should be extracted. Should be a list
            containing the ints 0, 1, or both, or None/the empty list
            to extract sky only.
            FIXME: Needs testing

        Returns
        -------
        profiles : list of :any:`numpy.ndarray`
            List of object slit profiles, as :any:`numpy.ndarray`.

        TODO: Figure out centroid array behaviour if needed.
        """
        # Input checking
        if used_objects is None:
            used_objects = []
        if len(used_objects) > 2:
            raise ValueError('used_objects must have length 1 or 2')
        used_objects = [int(_) for _ in used_objects]
        if not np.all([_ in [0, 1] for _ in used_objects]):
            raise ValueError('Only 0 and 1 may be in used_objects')
        if len(used_objects) != len(set(used_objects)):
            raise ValueError('Duplicate values are not allowed in '
                             'used_objects')

        # Find the slit profile.
        full_profile = self.slit_profile(arm=arm, reverse_profile=True)

        if correct_for_sky or append_sky:
            # Get the flat profile from the flat image.
            flat_profile = self.slit_profile(arm=arm, use_flat=True, reverse_profile=True)

        # WARNING: This is done in the extracted profile space. Is there any
        # benefit to doing this in pixel space? Maybe yes for the centroid.
        if correct_for_sky:
            flat_scaling = np.median(full_profile[
                self.sky_pix_only_boundaries[arm][0]:
                self.sky_pix_only_boundaries[arm][1] + 1
            ]) / np.median(flat_profile[
                self.sky_pix_only_boundaries[arm][0]:
                self.sky_pix_only_boundaries[arm][1] + 1
            ])
            full_profile -= flat_scaling*flat_profile

        # Extract the objects.
        profiles = []
        for boundary in [self.object_boundaries[arm][_] for _ in used_objects]:
            profiles.append(np.copy(full_profile))
            profiles[-1][:boundary[0]] = 0
            profiles[-1][boundary[1]+1:] = 0

        # Append the "sky" if needed (be aware that the top & bottom pixel
        # borders [the "edges" of the profile] contain some object flux)
        if append_sky:
            profiles.append(flat_profile)
            profiles[-1][:self.sky_pix_boundaries[arm][0]] = 0
            profiles[-1][self.sky_pix_boundaries[arm][1]+1:] = 0
        profiles = np.array(profiles)

        # Blank out any regions where the IFU is said to be stowed
        # Ensure we get pixels at the end of the slit that might not
        # be formally within the boundaries
        for stowed in self.stowed:
            boundary = self.object_boundaries[arm][stowed]
            if (stowed == 0) and (self.mode == 'std'):
                sky_only_boundary = self.sky_pix_only_boundaries[arm]
                profiles[:, :sky_only_boundary[1]+1] = 0
            else:
                profiles[:, boundary[0]:boundary[1]+1] = 0

        # Normalise profiles if requested (not needed if the total flux is what
        # you're after, e.g. for an mean exposure epoch calculation)
        if normalise_profiles:
            for prof in profiles:
                profsum = np.sum(prof)
                if math.isclose(profsum, 0.0, abs_tol=1e-9):
                    # FIXME what to do here?
                    # raise ZeroDivisionError('Sum of profile is too close to zero')
                    # CJS: normalization is only done for profile fitting for
                    # the extraction and if an IFU is stowed that's OK
                    pass
                else:
                    prof /= np.sum(prof)

        if self.reverse_profile[arm]:
            return profiles
        else:
            return profiles[:,::-1]

    def model_profile(self, slit_image=None, flat_image=None, rapid_fit=True):
        """
        Estimate seeing and create a model slit image.

        This method fits multiple predetermined fibre profiles in a
        uniformly-spaced linear manner and performs a least-squares fit to
        an (unrotated) slitviewer image. If a slitflat image is provided,
        a fit is first performed to that image, since the S/N will be good,
        and then the fibre positions are held fixed when fitting to the
        slit image. Otherwise, the four paramters that define the locations
        of the fibres are fit simultaneously with the slit image. If only
        flat_image is provide, then the models describing the flats will be
        returned.

        Parameters
        ----------
        slit_image: ndarray/None
            a slit image taken on-sky
        flat_image: ndarray/None
            a processed_slitflat image
        slitvpars: dict
            parameters from the appropriate SLITVMOD file
        binning: int
            binning of slitviewer image along each axis
        rapid_fit: bool
            hold all amplitudes in the slitflat equal to speed up the fit?
            (the slitflat is only used to determine fibre positions and these
            vary by ~0.01 pixels if this is set)

        Returns
        -------
        slit_model: dict
            FibreSlit model instances for the blue and red arms, keyed by arm
        """
        if slit_image is None and flat_image is None:
            raise ValueError("At least one of slit_image and flat_image "
                             "must be provided")

        model_class = FibreSlitStd if self.mode == "std" else FibreSlitHigh

        slit_models = {}
        for arm in ('blue', 'red'):
            _slice, center = self.get_raw_slit_position(arm)
            init_parameters = {'x_center': center[1] - _slice[1].start,
                               'y_center': center[0] - _slice[0].start,
                               'angle': self.rota}
            fit_it = fitting.LevMarLSQFitter()
            if flat_image is not None:
                slit_cutout = flat_image[_slice]
                slit_model = model_class(**init_parameters, shape=slit_cutout.shape)

                if rapid_fit:  # fix all the amplitudes to be equal
                    tied_amplitude = None
                    for p in slit_model.param_names:
                        if p.startswith('amplitude'):
                            if tied_amplitude is None:
                                tied_amplitude = p
                            else:
                                getattr(slit_model, p).tied = lambda model: getattr(model, tied_amplitude)

                slitflat_final = fit_it(slit_model, slit_cutout.flatten(),
                                        slit_cutout.flatten(), maxiter=400)
            else:
                slitflat_final = None

            if slit_image is None:
                slit_models[arm] = slitflat_final
            else:
                slit_cutout = slit_image[_slice]
                if self.mode == "std":
                    # Model IFU0 only on the basis that the brightest object is there
                    slit_model = model_class(
                        **init_parameters, shape=slit_cutout.shape, ifus=['ifu0'])
                else:
                    slit_model = model_class(
                        **init_parameters, shape=slit_cutout.shape, ifus=['ifu'])
                if flat_image is not None:
                    # Hold the fibre locations fixed when fitting the slit
                    for param in ('x_center', 'y_center', 'separation', 'angle'):
                        setattr(slit_model, param, getattr(slitflat_final, param))
                        slit_model.fixed[param] = True
                slit_final = fit_it(slit_model, slit_cutout.flatten(),
                                    slit_cutout.flatten(), maxiter=200)
                slit_models[arm] = slit_final
        return slit_models

    def get_raw_slit_position(self, arm):
        """
        Determine the rectangular slice that fully encloses the image of the
        slit for this arm, and the location of the slit center.

        Parameters
        ----------
        arm: str
            'red' or 'blue'

        Returns
        -------
        slice:
            the 2D slice containing the entire slit image
        (y, x):
            location of the slit center in the full image
        """
        y_halfwidth = int(self.slit_length / self.microns_pix / 2)

        recenter = (models.Shift(self.center[0]) & models.Shift(self.center[1]))
        to_raw = recenter.inverse | models.Rotation2D(self.rota) | recenter
        yc0, xc0 = self.central_pix[arm]
        xr, yr = to_raw([xc0, xc0, xc0], [yc0, yc0 - y_halfwidth,
                                          yc0 + y_halfwidth])
        x1, x2 = xr.min() - 8 / self.binning, xr.max() + 8 / self.binning
        y1, y2 = yr.min() - 8 / self.binning, yr.max() + 8 / self.binning
        _slice = (slice(int(y1), int(y2 + 1)), slice(int(x1), int(x2 + 1)))
        return _slice, (yr[0], xr[0])

    def create_image(self, shape=None, blue_model=None, red_model=None):
        """
        Create a synthetic slitviewer image based on models for the two
        pseudoslits.

        Parameters
        ----------
        shape: 2-tuple
            shape of the output image (includes binning)
        blue_model: FibreSlit instance
            model for the blue fibres
        red_model: FibreSlit instance
            model for the red fibres

        Returns
        -------
        image: 2D ndarray
            a synthetic slitviewer image
        """
        image = np.zeros(shape, dtype=np.float32)
        for arm, model in zip(('blue', 'red'), (blue_model, red_model)):
            if model is not None:  # add some flexibility
                _slice, center = self.get_raw_slit_position(arm)
                _shape = [s.stop - s.start for s in _slice]
                image[_slice] += model(np.zeros(_shape))
        return image

    def fake_slitimage(self, unbinned_shape=(260, 300), amplitude=10000,
                       flat_image=None, ifus=None, seeing=None, alpha=4):
        """
        Construct a fake slit (or slitflat) image and set the appropriate
        attribute to the rotated version of this image.

        This method will create and return a synthetic slitviewer image
        representing either a slit image or a slitflat image, depending on
        the parameters. Fibre positions will be determined based on the
        parameters in the SLITVMOD and the fibre-to-fibre separation in the
        appropriate FibreSlit model unless a flat_image is provided, in which
        case the positions will be derived from a fit to that image.

        A synthetic slit image is created if a seeing value (or values,
        as a dict) is provided, otherwise a synthetic slitflat is created.

        Parameters
        ----------
        unbinned_shape: 2-tuple/None
            shape of the image (y, x) if unbinned
            (None is valid if flat_image is provided)
        amplitude: float/tuple
            amplitude of brightest fibre in each arm
            (a tuple should be provided for 2-IFU mode when making a slit image)
        flat_image: 2D ndarray (fake slit only)
            create a slit image based on the fibre positions in this slitflat
        ifus: list/None
            ifus with signal
        seeing: float/dict/None
            FWHM in each arm (None => create a slitflat)
        alpha: float
            exponent of the Moffat profile (fake slit only)

        Returns
        -------
        image: 2D ndarray
            a synthetic image approximating a slit/slitflat image (unrotated)
        """
        if seeing is None and flat_image is not None:
            raise ValueError("A slitflat image has been provided but no "
                             "seeing estimate")

        if flat_image is None:
            shape = tuple(length // self.binning for length in unbinned_shape)
        else:
            shape = flat_image.shape
            slitflat_models = self.model_profile(flat_image=flat_image)
        model_class = FibreSlitStd if self.mode == "std" else FibreSlitHigh
        if not isinstance(seeing, dict) and seeing is not None:
            seeing = {'blue': seeing, 'red': seeing}
        slit_models = {}
        for arm in ('blue', 'red'):
            _slice, center = self.get_raw_slit_position(arm)
            _shape = tuple(s.stop - s.start for s in _slice)
            if flat_image is None:
                positional_params = {'x_center': center[1] - _slice[1].start,
                                     'y_center': center[0] - _slice[0].start,
                                     'angle': self.rota}
            else:
                positional_params = {k: getattr(slitflat_models[arm], k)
                                     for k in ('x_center', 'y_center',
                                               'angle', 'separation')}
            slit_model = model_class(**positional_params, ifus=ifus, shape=_shape)
            if seeing is None:  # making a synthetic sliflat
                amp_params = [param for param in slit_model.param_names
                              if 'amplitude' in param]
                fluxes = np.full(amplitude, len(slit_model.param_names)-4)
            else:  # making a synthetic slit
                if arm == 'blue':  # no need to create it twice!
                    ifu = slit_model.model_ifu()
                fluxes = slit_model.fibre_fluxes(ifu, fwhm=seeing[arm], alpha=alpha)
                # We take advantage of the behaviour of zip taking the
                # shortest list. So we can double the number of amplitudes
                # for std mode to replicate the second IFU and it doesn't
                # matter for high mode.
                try:
                    fluxes = np.tile(fluxes * amplitude / fluxes.max(), 2)
                except ValueError:  # it's got two values
                    fluxes = np.ravel([fluxes * amp / fluxes.max()
                                       for amp in amplitude])
                amp_params = [param for param in slit_model.param_names
                              if 'amplitude' in param and 'ifu' in param]
            for param, fflux in zip(amp_params, fluxes):
                    setattr(slit_model, param, fflux)
            slit_models[arm] = slit_model
        image = self.create_image(shape, blue_model=slit_models['blue'],
                                  red_model=slit_models['red'])
        if seeing is None:
            self.flat_image = transform.rotate(
                util.img_as_float64(image), self.rota, center=self.center)
        else:
            self.slit_image = transform.rotate(
                util.img_as_float64(image), self.rota, center=self.center)
        return image


class FibreSlit(Fittable1DModel):
    """
    Note: due to periodicity in the profile, the initial guess for the
    center *must* be nearer to the central fibre than any other fibre
    when trying to fit to a slitviewer image.
    """
    _param_names = ()

    pixels_per_arcsec = 100

    def __init__(self, name=None, meta=None, ifus=None,
                 subsample=5, shape=None, fibre_model=None, binning=2,
                 **params):
        ny, nx = shape
        self.shape = shape
        self.binning = binning
        ygrid, xgrid = np.mgrid[:ny*subsample, :nx*subsample]
        self.ygrid = (ygrid + 0.5) / subsample - 0.5
        self.xgrid = (xgrid + 0.5) / subsample - 0.5
        if fibre_model is None:
            self.fibre_model = self.default_fibre_model
        else:
            self.fibre_model = fibre_model
        if ifus is None:
            ifus = self.default_fibres
        self.fibres = {f"{k}_{i}": x for k, v in self.fibre_mapping.items()
                       for i, x in enumerate(np.arange(v[0], v[1]+0.01)) if k in ifus}
        self.subsample = subsample
        self._param_names = self._generate_param_names()
        for param_name in self._param_names:
            self._parameters_[param_name] = Parameter(
                param_name, default=np.array(
                    self.default_param_values.get(param_name, 0.)))
        super().__init__(name=name, meta=meta, **params)

    @property
    def param_names(self):
        return self._param_names

    def _generate_param_names(self):
        names = ['x_center', 'y_center', 'separation', 'angle']
        names.extend(f"amplitude_{k}" for k in self.fibres.keys())
        return tuple(names)

    def evaluate(self, x, x_center, y_center, separation, angle, *amplitudes):
        """x is simply a dummy array"""
        oversampled_array = np.zeros_like(self.xgrid)
        sep = separation / self.binning
        fibres = np.array(list(self.fibres.values()))
        xc_all = x_center - fibres * sep * np.sin(angle * np.pi / 180)
        yc_all = y_center + fibres * sep * np.cos(angle * np.pi / 180)
        for x0, y0, amp in zip(xc_all, yc_all, amplitudes):
            rsq = (self.xgrid - x0) ** 2 + (self.ygrid - y0) ** 2
            oversampled_array += amp * self.fibre_model(np.sqrt(rsq))

        out_array = oversampled_array.reshape(
            self.shape[0], self.subsample,
            self.shape[1], self.subsample).mean(axis=(1, -1))
        return out_array.flatten()

    def add_hexagon(self, image, x, y, size=None, value=0):
        """
        Add a hexagon to the IFU focal plane image

        Parameters
        ----------
        image: ndarray
            the IFU focal plane image to be modified
        x: float
            x-pixel location of hexagon centre
        y: float
            y-pixel location of hexagon centre
        size: float
            size of hexagon (distance from flat to flat)
        value: int
            value to insert into focal plane image
        """
        size /= 0.5 * np.sqrt(3)
        xv = image.shape[1] // 2 + x + size * np.cos(np.arange(6) * np.pi / 3)
        yv = image.shape[0] // 2 + y + size * np.sin(np.arange(6) * np.pi / 3)
        mask = draw.polygon2mask(image.shape, list(zip(yv, xv)))
        image[mask] = value
        return image

    def fibre_fluxes(self, ifu, fwhm=0.5, alpha=4):
        y0, x0 = [x // 2 for x in ifu.shape]
        gamma = fwhm / (2 * np.sqrt(2 ** (1/alpha) - 1))
        r = (np.sqrt(np.square(np.mgrid[-y0:y0+1, -x0:x0+1]).sum(axis=0)) /
             self.pixels_per_arcsec)
        moffat = models.Moffat1D(gamma=gamma, alpha=alpha)(r)
        ifu_fluxes = np.array([moffat[ifu==fibre].sum()
                               for fibre in range(ifu.max()+1)])
        total_flux = self.pixels_per_arcsec ** 2 * np.pi * gamma**2 / (alpha - 1)
        return ifu_fluxes / total_flux

    def estimate_seeing(self):
        """Estimate the seeing and flux loss from the modelled fibre fluxes"""
        amplitudes = np.asarray(self.parameters)[4:]
        ifu = self.model_ifu()
        func = lambda x: x[0] * self.fibre_fluxes(ifu, fwhm=x[1]) - amplitudes
        result = least_squares(func, [3*amplitudes.max(), 0.5], bounds=(0, np.inf))
        frac_light = self.fibre_fluxes(ifu, fwhm=result.x[1]).sum()
        return result.x[1], frac_light


class FibreSlitHigh(FibreSlit):
    """Things specific to the high-resolution SlitView object"""
    default_param_values = {'angle': 81.,
                            'separation': 7.38}

    fibre_mapping = {'ifu': (-6.5, 11.5),
                     'sky': (-13.5, -7.5),
                     'cal': (13.5, 13.5)}

    fibre_pitch = 0.144 * 1.611444  # arcsec

    default_fibres = ['ifu', 'sky']  # not the ThXe calibration source

    # determined empirically
    def default_fibre_model(self, r):
        return np.exp(-(r * self.binning / 3.42) ** 3.31)

    def model_ifu(self):
        """
        Create an image of the IFU focal plane with each data value
        representing a different fibre.

        Returns
        -------
        ifu: 2D ndarray
            image representing the focal plane
        """
        xc = int(0.8 * self.pixels_per_arcsec)
        ifu = np.full((2*xc+1, 2*xc+1), -1, dtype=int)
        pitch = self.fibre_pitch * self.pixels_per_arcsec
        size = 0.5 * 0.94 * pitch  # reproduces Fig.2.2.3a of GHOSD-09
        S3 = np.sqrt(3)
        self.add_hexagon(ifu, 0, 0, size, 9)
        for i, (fibre1, fibre2, fibre3) in enumerate(zip((6, 12, 10, 7, 11, 8),
                                                         (3, 14, 5, 13, 15, 4),
                                                         (0, 18, 16, 2, 17, 1))):
            self.add_hexagon(ifu, pitch * np.sin(i * np.pi / 3),
                             pitch * np.cos(i * np.pi / 3), size, fibre1)
            self.add_hexagon(ifu, S3 * pitch * np.sin((i + 0.5) * np.pi / 3),
                             S3 * pitch * np.cos((i + 0.5) * np.pi / 3), size, fibre2)
            self.add_hexagon(ifu, 2 * pitch * np.sin(i * np.pi / 3),
                             2 * pitch * np.cos(i * np.pi / 3), size, fibre3)
        return ifu


class FibreSlitStd(FibreSlit):
    """Things specific to the std-resolution SlitView object"""
    default_param_values = {'angle': 81.,
                            'separation': 12.2}

    fibre_mapping = {'ifu0': (2, 8),
                     'ifu1': (-8, -2),
                     'sky': (-1, 1)}

    fibre_pitch = 0.240 * 1.611444  # arcsec

    default_fibres=['ifu0', 'ifu1', 'sky']

    # determined empirically
    def default_fibre_model(self, r):
        return np.exp(-(r * self.binning / 5.46) ** 3.31)

    def model_ifu(self):
        """
        Create an image of the IFU focal plane with each data value
        representing a different fibre.

        Returns
        -------
        ifu: 2D ndarray
            image representing the focal plane
        """
        xc = int(0.8 * self.pixels_per_arcsec)
        ifu = np.full((2*xc+1, 2*xc+1), -1, dtype=int)
        pitch = self.fibre_pitch * self.pixels_per_arcsec
        size = 0.5 * 0.97 * pitch  # reproduces Fig.2.2.3a of GHOSD-09
        self.add_hexagon(ifu, 0, 0, size, 3)
        for i, fibre in enumerate((0, 6, 4, 1, 5, 2)):
            self.add_hexagon(ifu, pitch * np.sin(i * np.pi / 3),
                             pitch * np.cos(i * np.pi / 3), size, fibre)
        return ifu
