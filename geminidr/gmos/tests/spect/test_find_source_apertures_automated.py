from astropy.io import fits as pf
import numpy as np
from astropy.modeling.models import Gaussian1D
from itertools import product as cart_product

SHAPE = (2048, 200)
RDNOISE = 4

# Edit these to provide iterables to iterate over
BACKGROUNDS = (0,)  # overall background level
PEAKS = (50,)  # signal in galaxy peak
CONTRASTS = (0.25,)  # ratio of SN peak to galaxy peak
SEPARATIONS = (8,)  # pixel separation between galaxy/SN peaks
GAL_FWHMS = (40,)  # FWHM (pixels) of galaxy
SN_FWHMS = (3,)  # FWHM (pixels) of SN

phu_dict = dict(
    INSTRUME= 'GMOS-N',
    OBJECT  = 'FAKE',
    OBSTYPE = 'OBJECT',
)
hdr_dict = dict(
    WCSAXES =                    3,
    WCSDIM  =                    3,
    CD1_1   =  -0.1035051453281131,
    CD2_1   =                  0.0,
    CD1_2   =                  0.0,
    CD2_2   = -1.6608167576414E-05,
    CD3_2   = 4.17864941757412E-05,
    CD1_3   =                  0.0,
    CD2_3   =                  0.0,
    CD3_3   =                  1.0,
    CRVAL2  =     76.3775200034826,
    CRVAL3  =     52.8303306863311,
    CRVAL1  =                495.0,
    CTYPE1  = 'AWAV    ',
    CTYPE2  = 'RA---TAN',
    CTYPE3  = 'DEC--TAN',
    CRPIX1  =    1575.215466689882,
    CRPIX2  =   -555.7218408956066,
    CRPIX3  =                  0.0,
    CUNIT1  = 'nm      ',
    CUNIT2  = 'deg     ',
    CUNIT3  = 'deg     ',
    DATASEC = '[1:{1},1:{0}]'.format(*SHAPE),
)

yc = 0.5 * SHAPE[0]
for bkgd, peak, contrast, sep, gal_fwhm, sn_fwhm in cart_product(
        BACKGROUNDS, PEAKS, CONTRASTS, SEPARATIONS, GAL_FWHMS, SN_FWHMS):
    gal_std = 0.42466 * gal_fwhm
    sn_std = 0.42466 * sn_fwhm
    model = (Gaussian1D(amplitude=peak, mean=yc, stddev=gal_std) +
             Gaussian1D(amplitude=peak*contrast, mean=yc+sep, stddev=sn_std))
    profile = model(np.arange(SHAPE[0]))
    print(profile[:5])
    data = np.zeros(SHAPE) + profile[:, np.newaxis]
    data += np.random.normal(scale=RDNOISE, size=data.size).reshape(data.shape)

    hdulist = pf.HDUList([pf.PrimaryHDU(header=pf.Header(phu_dict)),
                          pf.ImageHDU(data=data, header=pf.Header(hdr_dict))])
    filename = (f"fake_bkgd{bkgd:04.0f}_peak{peak:03.0f}_con{contrast:4.2f}_"
                f"sep{sep:05.2f}_gal{gal_fwhm:5.2f}_sn{sn_fwhm:4.2f}.fits")
    hdulist.writeto(filename, overwrite=True)


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad_center_tolerance_snr", extra_test_data, indirect=True)
def test_find_apertures_generated(ad_center_tolerance_snr):
    """
    Test that p.findApertures can find apertures in special test cases, such as
    with galaxy background
    """
    ad, expected_center, range, snr, count = ad_center_tolerance_snr
    args = dict()
    if snr is not None:
        args["min_snr"] = snr
    p = GMOSSpect([ad])
    _ad = p.findApertures(max_apertures=1, **args).pop()

    assert hasattr(ad[0], 'APERTURE')
    if count is not None:
        assert(len(ad[0].APERTURE) == count)
    if expected_center is not None:
        assert len([ext for ext in ad if abs(ext.APERTURE['c0'] - expected_center) < range]) >= 1
