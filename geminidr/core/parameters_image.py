# This parameter file contains the parameters related to the primitives located
# in the primitives_image.py file, in alphabetical order.
import numpy as np
from gempy.library import config
from astrodata import AstroData
from geminidr.core import parameters_stack, parameters_photometry


class fringeCorrectConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_fringeCorrected", optional=True)
    fringe = config.ListField("Fringe frame to subtract", (str, AstroData),
                              None, optional=True, single=True)
    do_fringe = config.Field("Perform fringe correction?", bool, None, optional=True)
    scale = config.Field("Scale fringe frame?", bool, None, optional=True)
    scale_factor = config.ListField("Scale factor for fringe frame", float, None,
                                    optional=True, single=True)


class makeFringeFrameConfig(parameters_stack.core_stacking_config,
                            parameters_photometry.detectSourcesConfig):
    subtract_median_image = config.Field("Subtract median image?", bool, True)
    dilation = config.RangeField("Object dilation radius (pixels)", float, 2., min=0)

    def setDefaults(self):
        self.suffix = "_fringe"


class makeFringeForQAConfig(makeFringeFrameConfig):
    pass


class scaleByIntensityConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_scaled", optional=True)
    section = config.Field("Statistics section", str, None, optional=True)
    scaling = config.ChoiceField("Statistic for scaling", str,
                                 allowed={"mean": "Scale by mean",
                                          "median": "Scale by median"},
                                 default="mean")
    separate_ext = config.Field("Scale extensions separately?", bool, False)


class flagCosmicRaysByStackingConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_CRMasked", optional=True)
    hsigma = config.RangeField("High rejection threshold (sigma)", float, 7., min=0)
    dilation = config.RangeField("CR dilation radius (pixels)", float, 1., min=0)


class flagCosmicRaysConfig(config.Config):
    suffix = config.Field(
        doc="Filename suffix",
        dtype=str,
        default="_CRMasked",
        optional=True,
    )
    sigclip = config.Field(
        doc="Laplacian-to-noise limit for cosmic ray detection. Lower "
        "values will flag more pixels as cosmic rays.",
        dtype=float,
        optional=True,
        default=4.5,
    )
    sigfrac = config.Field(
        doc="Fractional detection limit for neighboring pixels. For cosmic "
        "ray neighbor pixels, a lapacian-to-noise detection limit of"
        "sigfrac * sigclip will be used.",
        dtype=float,
        optional=True,
        default=0.3,
    )
    objlim = config.Field(
        doc="Minimum contrast between Laplacian image and the fine structure "
        "image.  Increase this value if cores of bright stars are flagged as "
        "cosmic rays.",
        dtype=float,
        optional=True,
        default=5.0,
    )
    pssl = config.Field(
        doc="Previously subtracted sky level in ADU. We always need to work "
        "in electrons for cosmic ray detection, so we need to know the sky "
        "level that has been subtracted so we can add it back in.",
        dtype=float,
        optional=True,
        default=0.0,
    )
    # gain = config.Field(
    #     doc="Gain of the image (electrons / ADU). We always need to work in "
    #     "electrons for cosmic ray detection.",
    #     dtype=float,
    #     optional=True,
    #     default=1.0,
    # )
    # readnoise = config.Field(
    #     doc="Read noise of the image (electrons). Used to generate the noise "
    #     "model of the image.",
    #     dtype=float,
    #     optional=True,
    #     default=6.5,
    # )
    # satlevel = config.Field(
    #     doc="Saturation of level of the image (electrons). This value is "
    #     "used to detect saturated stars and pixels at or above this level "
    #     "are added to the mask.",
    #     dtype=float,
    #     optional=True,
    #     default=65536.0,
    # )
    niter = config.Field(
        doc="Number of iterations of the LA Cosmic algorithm to perform",
        dtype=int,
        optional=True,
        default=4,
    )
    sepmed = config.Field(
        doc="Use the separable median filter instead of the full median "
        "filter. The separable median is not identical to the full median "
        "filter, but they are approximately the same and the separable median "
        "filter is significantly faster and still detects cosmic rays well.",
        dtype=bool,
        optional=True,
        default=True,
    )
    cleantype = config.ChoiceField(
        doc="Set which clean algorithm is used.",
        allowed={
            'median': 'An umasked 5x5 median filter',
            'medmask': 'A masked 5x5 median filter',
            'meanmask': 'A masked 5x5 mean filter',
            'idw': 'A masked 5x5 inverse distance weighted interpolation',
        },
        dtype=str,
        optional=True,
        default="meanmask",
    )
    fsmode = config.ChoiceField(
        doc="Method to build the fine structure image.",
        allowed={
            'median': 'Use the median filter in the standard LA Cosmic '
            'algorithm',
            'convolve': 'Convolve the image with the psf kernel to calculate '
            'the fine structure image.',
        },
        dtype=str,
        optional=True,
        default='median',
    )
    psfmodel = config.ChoiceField(
        doc="Model to use to generate the psf kernel if fsmode == 'convolve' "
        "and psfk is None. The current choices are Gaussian and Moffat "
        "profiles.",
        allowed={
            'gauss': 'Circular Gaussian kernel',
            'moffat': 'Circular Moffat kernel',
            'gaussx': 'Gaussian kernel in the x direction',
            'gaussy': 'Gaussian kernel in the y direction',
        },
        dtype=str,
        optional=True,
        default="gauss",
    )
    psffwhm = config.Field(
        doc="Full Width Half Maximum of the PSF to use for the kernel.",
        dtype=float,
        optional=True,
        default=2.5,
    )
    psfsize = config.Field(
        doc="Size of the kernel to calculate. Returned kernel will have size "
        "psfsize x psfsize. psfsize should be odd.",
        dtype=int,
        optional=True,
        default=7,
    )
    # ndarray is not supported as dtype, and anyway it seems unprobable
    # that we want to pass a kernel array?
    # psfk = config.Field(
    #     doc="PSF kernel array to use for the fine structure image if "
    #     "fsmode == 'convolve'. If None and fsmode == 'convolve', we "
    #     "calculate the psf kernel using 'psfmodel'.",
    #     dtype=np.ndarray,
    #     optional=True,
    #     default=7,
    # )
    psfbeta = config.Field(
        doc="Moffat beta parameter. Only used if psfmodel=='moffat'.",
        dtype=float,
        optional=True,
        default=4.765,
    )
    verbose = config.Field(
        doc="Print to the screen or not.",
        dtype=bool,
        optional=True,
        default=False,
    )
