"""
Simulate GHOST/Veloce instrument observations.

This is a simple simulation code for GHOST or Veloce,
with a class :class:`GhostArm` that simulates
a single arm of the instrument. The key default parameters
are hardwired for each named
configuration in the :func:`__init__ <GhostArm.__init__>` function of ARM.

Note that in this simulation code, the 'x' and 'y' directions are the
along-slit and dispersion directions respectively
(similar to physical axes), but by convention, images are returned/displayed
with a vertical slit and a horizontal dispersion direction.

For a simple simulation, run, e.g.::

    import pymfe
    blue = pymfe.ghost.Arm('blue')
    blue.simulate_frame()

TODO:

1) Add spectrograph aberrations (just focus and coma)
2) Add pupil illumination plus aberrations.
"""
from __future__ import division, print_function
import numpy as np
from .polyspect import Polyspect

GHOST_BLUE_SZX = 4112 # 4096 # 
GHOST_BLUE_SZY = 4096 # 4112 # 
GHOST_RED_SZX = 6160
GHOST_RED_SZY = 6144


class GhostArm(Polyspect):
    """
    Class representing an arm of the spectrograph.

    A class for each arm of the spectrograph. The initialisation
    function takes a series of strings representing the configuration.
    It can be ``"red"`` or ``"blue"`` for the arm (first string),
    and ``"std"`` or ``"high"`` for the mode (second string).

    This class initialises and inherits all attributes and
    methods from :any:`Polyspect`, which is the module that
    contains all spectrograph generic functions.
    """

    def __init__(self, arm='blue', mode='std',
                 detector_x_bin=1, detector_y_bin=1):
        """
        The class initialisation takes the arm, resolution mode and binning
        modes as inputs and defines all needed attributes.

        It starts by initialising the :any:`Polyspect` class with the correct
        detector sizes ``szx`` and ``szy``, order numbers (``m_min``, ``m_max``)
        and whether the CCD is transposed. Transposed in this case implies that
        the spectral direction is in the x axis of the CCD image, which is the
        case for the GHOST data.

        Most of the parameters sent to the
        :class:`PolySpect <polyfit.polyspect.PolySpect>` initialization function
        are self-explanatory, but here is a list of those
        that may not be:

        +------------------------------+---------------------------------------+
        | **Variable Name**            | **Purpose/meaning**                   |
        +------------------------------+---------------------------------------+
        | ``m_ref``, ``m_min`` and     | Reference, minimum and maximum order  |
        | ``m_max``                    | indices for the camera.               |
        +------------------------------+---------------------------------------+
        | ``szx`` and ``szy``          | Number of pixels in the x and y       |
        |                              | directions                            |
        +------------------------------+---------------------------------------+
        | ``nlenslets``                | Number of lenslets in the IFU         |
        +------------------------------+---------------------------------------+
        | ``lenslet_high_size`` and    | Unused                                |
        | ``lenslet_std_size``         |                                       |
        +------------------------------+---------------------------------------+
        
        Attributes
        ----------
        arm: str
            Which arm of the GHOST spectrograph is to be initialized. Can be
            ``'red'`` or ``'blue'``.
        spect: str
            Which spectrograph in usage. Defaults to ``'ghost'``.
        lenslet_high_size: int
            Lenslet flat-to-flat in microns for high mode. Defaults to
            ``118.0```.
        lenslet_std_size: int
            Lenslet flat-to-flat in microns for standard mode. Defaults to
            ``197.0``.
        mode: str
            Resolution mode. Can be either ``'std'`` or ``'high'``.
        nlenslets: int
            Number of lenslets of the IFU. This value is set depending on
            whether ``mode`` is ``'std'`` (17) or ``'high'`` (28).
        detector_x_bin: int, optional
            The x binning of the detector. Defaults to 1.
        detector_y_bin: int, optional
            The y binning of the detector. Defaults to 1.
        """
        # MCW 190822 - swapped szy and szx values for new data
        if arm == 'red':
            Polyspect.__init__(self, m_ref=50,
                               szx=GHOST_RED_SZX, szy=GHOST_RED_SZY,
                               m_min=33, m_max=65, transpose=True)
        elif arm == 'blue':
            Polyspect.__init__(self, m_ref=80,
                               szx=GHOST_BLUE_SZX, szy=GHOST_BLUE_SZY,
                               m_min=64, m_max=98, transpose=True) #was 63 and 95, or 62 and 92 for shifted NRC data.
        else:
            print("Unknown spectrograph arm!")
            raise UserWarning
        # A lot of these parameters are yet unused.
        # These are a legacy of the original simulator and are left here
        # because they may become useful in the future.
        self.spect = 'ghost'
        self.arm = arm
        self.lenslet_high_size = 118.0  # Lenslet flat-to-flat in microns
        self.lenslet_std_size = 197.0  # Lenslet flat-to-flat in microns
        self.mode = mode
        # x is in the spatial direction for this module, contrary to intuition
        # and convention because Mike is weird like this....
        # y is in the spectral direction
        self.ybin, self.xbin = detector_x_bin, detector_y_bin

        # Now we determine the number of fibers based on mode.
        if mode == 'high':
            self.nlenslets = 28
        elif mode == 'std':
            self.nlenslets = 17
        else:
            print("Unknown mode!")
            raise UserWarning

    def bin_data(self, data):
        """
        Generic data binning function.

        Generic function used to create a binned equivalent of a
        spectrograph image array for the purposes of equivalent extraction.
        Data are binned to the binning specified by the class attributes
        ``detector_x_bin`` and ``detector_y_bin``.

        This function is mostly used to re-bin calibration images (flat,
        arc, etc) to the binning of related science data prior to performing
        extraction.

        .. note::
            This function is now
            implemented elsewhere as the
            :any:`ghost.primitives_ghost.GHOST._rebin_ghost_ad`
            method in the :any:`ghost.primitives_ghost.GHOST` primtive class,
            and takes care of all the binning.

        Parameters
        ----------
        data: :obj:`numpy.ndarray`
            The (unbinned) data to be binned
        
        Raises
        ------
        UserWarning
            If the data provided is not consistent with CCD size, i.e., not
            unbinned

        Returns
        -------
        binned_array: :obj:`numpy.ndarray`
            Re-binned data.
        """
        if data.shape != (self.szx, self.szy):
            raise UserWarning('Input data for binning is not in the expected'
                              'format')

        if self.xbin == 1 and self.ybin == 1:
            return data

        rows = self.xbin
        cols = self.ybin
        binned_array = data.reshape(int(data.shape[0] / rows),
                                    rows,
                                    int(data.shape[1] / cols),
                                    cols).sum(axis=1).sum(axis=2)
        return binned_array

    def slit_flat_convolve(self, flat, slit_profile=None, spatpars=None,
                           microns_pix=None, xpars=None, num_conv=3):
        """
        Correlate a flat field image with a slit profile image. Note that this is
        not convolution, as we don't want to shift the image.

        Function that takes a flat field image and a slit profile and
        convolves them in two dimensions. Returns result of the convolution,
        which should be used for tramline fitting in findApertures.

        This function is a first step in finding the centre of each order.
        Given the potentially overlapping nature of fiber images in flat field
        frames, a convolution method is employed with a sampled slit profile,
        in which the center of the order will, ideally, match the profile best
        and reveal the maximum of the convolution.

        A convolution map is then fed into a fitting function where the location
        of the maxima in the map are found, and a model is fit to determine a
        continuous function describing the centre of the orders.

        Unfortunately, the slit magnification changes across the detector. Rather
        than writing a giant for loop, num_conv convolutions are performed with
        a different slit magnification corresponding to each order stored in the
        list orders.

        For each of these orders, a convolution is done in 2D by interpolating 
        the magnified slit profile with the slit coordinates, normalising it and 
        inverse Fourier transforming the product between the flat transform and the 
        shifted slit profile::

          # Create the slit model.
          mod_slit = np.interp(profilex*spat_scale[i], slit_coord, slit_profile)

          # Normalise the slit model and Fourier transform for convolution
          mod_slit /= np.sum(mod_slit)
          mod_slit_ft = np.fft.rfft(np.fft.fftshift(mod_slit))

          flat_conv_cube[j, :, i] = np.fft.irfft((im_fft[:, i] * mod_slit_ft)/num_conv)

        Now, we have the convolution at ``num_conv`` orders, and the final
        result is an interpolation between these.

        .. note::

            The function currently contains code related to convoling with a
            fixed synthetic profile, which is not used. This is legacy code and
            sometimes used only for testing purposes. The normal usage is to
            have the spatial scale model parameters as inputs which determine
            the slit magnification as a function of order and pixel along the
            orders. The flat convolution is done using only a smaller number of
            orders (defaults to 3) and interpolated over the others but could in
            principle be done with all orders considered.

        Parameters
        ----------
        flat: :obj:`numpy.ndarray`
            A flat field image from the spectrograph
            
        slit_profile: :obj:`numpy.ndarray`, optional
            A slit profile as a 1D array with the slit profile fiber amplitudes.
            If none is supplied this function will assume identical fibers and
            create one to be used in the convolution based on default parameters
            specified in the ghost class.
            
        spatpars: :obj:`numpy.ndarray`, optional
            The 2D polynomial parameters for the slit spatial scale. 
            Required if slit_profile is not None.
        
        microns_pix: float, optional
            The slit scale in microns per pixel.
            Required if slit_profile is not None.
            
        xpars:  :obj:`numpy.ndarray`, optional
            The 2D polynomial parameters for the x (along-slit) coordinate.
            Required if slit_profile is not None.
            
        num_conv: int, optional, optional
            The number of different convolution functions to use for different
            orders.
            The final convolved profile is an interpolation between these.

        Returns
        -------
        flat_conv: :obj:`numpy.ndarray`
            The convolved 2D array.
        """
        # FIXME: Error checking of inputs is needed here
        # TODO: Based on test of speed, the convolution code with an input
        #  slit_profile could go order-by-order.

        if self.arm == 'red':
            # Now put in the default fiber profile parameters for each mode.
            # These are used by the convolution function on polyspect
            # These were determined based on visual correspondence with
            # simulated data and may need to be revised once we have real
            # data. The same applies to the blue arm parameters.
            if self.mode == 'std':
                fiber_separation = 4.15
                profile_sigma = 1.1
            elif self.mode == 'high':
                fiber_separation = 2.49
                profile_sigma = 0.7
        elif self.arm == 'blue':
            # Additional slit rotation accross an order needed to match Zemax.
            # self.extra_rot = 2.0

            # Now put in the default fiber profile parameters for each mode.
            # These are used by the convolution function on polyspect
            if self.mode == 'std':
                fiber_separation = 3.97
                profile_sigma = 1.1
            elif self.mode == 'high':
                fiber_separation = 2.53
                profile_sigma = 0.7
        else:
            print("Unknown spectrograph arm!")
            raise UserWarning

        # Fourier transform the flat for convolution
        im_fft = np.fft.rfft(flat, axis=0)

        # Create a x baseline for convolution 
        xbase = flat.shape[0]
        profilex = np.arange(xbase) - xbase // 2

        # This is the original code which is based on the fixed fiber_separation
        # defined above.
        if slit_profile is None:
            flat_conv = np.zeros_like(im_fft)

            # At this point create a slit profile

            # Now create a model of the slit profile
            mod_slit = np.zeros(xbase)
            if self.mode == 'high':
                nfibers = 26
            else:
                nfibers = self.nlenslets

            for i in range(-(nfibers // 2), -(nfibers // 2) + nfibers):
                mod_slit += np.exp(-(profilex - i * fiber_separation) ** 2 /
                                   2.0 / profile_sigma ** 2)
            # Normalise the slit model and fourier transform for convolution
            mod_slit /= np.sum(mod_slit)
            mod_slit_ft = np.fft.rfft(np.fft.fftshift(mod_slit))

            # Now convolved in 2D
            for i in range(im_fft.shape[1]):
                flat_conv[:, i] = im_fft[:, i] * mod_slit_ft

            # Now inverse transform.
            flat_conv = np.fft.irfft(flat_conv, axis=0)

        # If a profile is given, do this instead.
        else:
            slit_profile_cor = slit_profile.copy()
            
            flat_conv = np.zeros_like(flat)
            flat_conv_cube = np.zeros((num_conv, flat.shape[0], flat.shape[1]))

            # Our orders that we'll evaluate the spatial scale at:
            orders = np.linspace(self.m_min, self.m_max, num_conv).astype(int)
            mprimes = self.m_ref / orders - 1

            # The slit coordinate in microns
            slit_coord = (np.arange(len(slit_profile)) -
                          len(slit_profile) / 2 + 0.5) * microns_pix

            x_map = np.empty((len(mprimes), self.szy))

            # Now convolved in 2D
            for j, mprime in enumerate(mprimes):
                # The spatial scales
                spat_scale = self.evaluate_poly(spatpars)[
                    orders[j] - self.m_min]

                # The x pixel values, just for this order
                x_map[j] = self.evaluate_poly(xpars)[orders[j] - self.m_min]
                
                for i in range(im_fft.shape[1]):
                    #CRH 20220901 Old Create the slit model.
                    #mod_slit = np.interp(profilex * spat_scale[i], slit_coord,
                    #                     slit_profile, left=0, right=0)

                    # Create the slit model and convolve it to the detector pixels
                    #n_slit_sample = int(np.round(spat_scale[i]/microns_pix))
                    #profilex_sample = np.arange(xbase*n_slit_sample)/n_slit_sample - xbase // 2

                    #mod_slit_sample = np.interp(profilex_sample * spat_scale[i],
                    #    slit_coord, slit_profile, left=0, right=0)
                    #mod_slit = mod_slit_sample.reshape(xbase,n_slit_sample).sum(axis=1)

                    from .extract import resample_slit_profiles_to_detector
                    # This will always have an odd number of pixels with the
                    # central one being the middle of the slit
                    mod_slit2 = resample_slit_profiles_to_detector(
                        [slit_profile], profile_y_microns=slit_coord,
                        profile_center=0, detpix_microns=spat_scale[i])[1][0]
                    #mod_slit2_ft = (np.fft.rfft(np.fft.fftshift(mod_slit2), n=flat.shape[0]))

                    # Normalise the slit model and Fourier transform for
                    # convolution. This has to be an l2 norm, in order to 
                    # work with variable slit lengths and mean that 
                    # the correlation peak is the least squares fit.
                    #mod_slit /= np.sum(mod_slit)
                    #mod_slit /= np.sqrt(np.sum(mod_slit ** 2))
                    #mod_slit_ft = np.fft.rfft(np.fft.fftshift(mod_slit))

                    # FIXME: Remove num_conv on next line and see if it makes
                    # a difference!
                    #flat_conv_cube[j, :, i] = np.fft.irfft(
                    #    (im_fft[:, i] * mod_slit_ft.conj()) / num_conv
                    #)
                    # CJS: Non-Fourier convolution
                    conv = np.correlate(flat[:, i], mod_slit2, mode="same")
                    flat_conv_cube[j, :, i] = conv / conv.sum()
                    #flat_conv_cube[j, :, i] = np.fft.irfft(
                    #    im_fft[:, i] * mod_slit2_ft.conj()) / num_conv

                    #from matplotlib import pyplot as plt
                    #if i + j == 0:
                    #    plt.ioff()
                    #    fig, ax = plt.subplots()
                    #    ax.plot(flat[:, i] / flat[:, i].max())
                    #    ax.plot(flat_conv_cube[j, :, i] / flat_conv_cube[j, :, i].max())
                    #    #ax.plot(conv)
                    #    #ax.plot(mod_slit2)
                    #    plt.show()

            # Work through every y coordinate and interpolate between the
            # convolutions with the different slit profiles.
            x_ix = np.arange(flat.shape[0]) - flat.shape[0] // 2

            # Create an m index, and reverse x_map if needed.
            # FIXME: This assumes a minimum size of x_map which should be
            # checked above, i.e. mprimes has 2 or more elements.
            m_map_ix = np.arange(len(mprimes))
            if x_map[1, 0] < x_map[0, 0]:
                m_map_ix = m_map_ix[::-1]
                x_map = x_map[::-1]
            for i in range(im_fft.shape[1]):
                m_ix_for_interp = np.interp(x_ix, x_map[:, i], m_map_ix)
                m_ix_for_interp = np.minimum(m_ix_for_interp,
                                             len(mprimes) - 1 - 1e-6)
                m_ix_for_interp = np.maximum(m_ix_for_interp, 0)
                m_ix_lo = np.int16(m_ix_for_interp)
                m_ix_hi = m_ix_lo + 1
                m_ix_frac = m_ix_for_interp - m_ix_lo
                for j in range(len(mprimes)):
                    weight = (m_ix_lo == j) * (1 - m_ix_frac) + (
                            m_ix_hi == j) * m_ix_frac
                    flat_conv[:, i] += weight * flat_conv_cube[j, :, i]

        return flat_conv
