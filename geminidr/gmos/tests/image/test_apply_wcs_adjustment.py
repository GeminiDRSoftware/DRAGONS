import pytest
import numpy as np

from astropy.table import Table
from astropy.modeling import models

import astrodata, gemini_instruments
from geminidr.gmos.primitives_gmos_image import GMOSImage
from gempy.library import transform, astromodels as am
from geminidr.gmos.lookups import geometry_conf as geotable

X = np.arange(-1500., 1501., 250.)
Y = np.full_like(X, 100.)

MODELS = (models.Shift(0) & models.Shift(2000),
          models.Rotation2D(2) | (models.Shift(0) & models.Shift(1500)),
          am.Scale2D(1.01) | models.Rotation2D(3) |
          (models.Shift(0) & models.Shift(100)))


@pytest.mark.parametrize("m_error", MODELS)
@pytest.mark.parametrize("mosaic", (False, True))
def test_apply_wcs_adjustment(m_error, mosaic, gmos_tiled_images):
    """
    This is quite a complex test so here's a thorough explanation.

    We create 4 GMOS images:
    AD1 and AD2 will undergo processing: AD1 has the correct WCS, while
    AD2 has some sort of error.
    AD3 and AD4 are versions of these images with their true WCS solutions
    attached.

    We create some object locations in (RA, DEC) and populate the OBJCATs
    of AD1/2, using their *true* WCS solutions (which are encoded in AD3/4).

    We then split AD1/2 into 3 streams (one for each CCD) and either:
    a) correct the WCS of AD2:CCD2 as if we had put the CCD2 stream through
       adjustWCSToReference()
    b) mosaic the "main" stream and correct the WCS of AD2's mosaic as if we
       had put the mosaics through adjustWCSToReference()

    We then run the CCD1 stream through applyWCSAdjustment() with the CCD2
    stream as reference. We confirm that the WCS of AD2:CCD1 is now correct by
    using it to determine the (RA, DEC) of its OBJCAT and comparing it to the
    (RA, DEC) of the OBJCAT from AD1:CCD1. We do the same for CCD3, and CCD2
    if we had mosaicked the images.
    """
    ad1, ad2, ad3, ad4 = gmos_tiled_images

    # Make AD4's WCS correct, incorporating the error, and then attach the
    # mosaic transform to AD3 and AD4 so CCDs 1 and 3 are correct wrt CCD2
    ad4[1].wcs.insert_transform(ad4[1].wcs.input_frame, m_error.inverse,
                                after=True)
    ad3 = transform.add_mosaic_wcs(ad3, geotable)
    ad4 = transform.add_mosaic_wcs(ad4, geotable)
    X0 = ad3[1].shape[1] // 2

    ra, dec = ad3[1].wcs(X + X0 - 1, Y - 1)
    # Place these objects in the OBJCAT of whichever extension they will
    # appear on. We use AD3/4 to determine the true locations (since they have
    # the true WCS solutions), but put them in the OBJCATs of AD1/2.
    for ad, ad_ref in zip((ad1, ad2), (ad3, ad4)):
        for ext, ext_ref in zip(ad, ad_ref):
            x, y = np.asarray(ext_ref.wcs.backward_transform(ra, dec))
            for xx, yy in zip(x, y):
                if 0 < xx < ext.shape[1] and 0 < yy < ext.shape[0]:
                    if hasattr(ext, "OBJCAT"):
                        ext.OBJCAT.add_row([len(ext.OBJCAT) + 1, xx + 1, yy + 1])
                    else:
                        ext.OBJCAT = Table([[1], [xx + 1], [yy + 1]],
                                           names=("NUMBER", "X_IMAGE", "Y_IMAGE"))

    p = GMOSImage([ad1, ad2])
    p.sliceIntoStreams(root_stream_name='ccd', copy=mosaic)
    if mosaic:
        p.mosaicDetectors()
        ref_stream = 'main'
    else:
        ref_stream = 'ccd2'

    # Mimic behaviour of adjustWCSToReference() by modifying second image's WCS
    wcs = p.streams[ref_stream][1][0].wcs
    if mosaic:
        # (1058,0) is the (x,y) offset from the mosaic to CCD2, since m_error
        # is in the frame of CCD2, so this is equivalent to the transform that
        # adjustWCSToReference() will obtain for the mosaic
        m_fix = models.Shift(-1058) & models.Shift(0) | m_error.inverse | (
            models.Shift(1058) & models.Shift(0))
    else:
        m_fix = m_error.inverse
    wcs.insert_transform(wcs.input_frame, m_fix, after=True)

    for stream in list({'ccd1', 'ccd2', 'ccd3'} - {ref_stream}):
        p.applyWCSAdjustment(stream=stream, reference_stream=ref_stream)
        ad1, ad2 = p.streams[stream]
        ra1, dec1 = ad1[0].wcs(ad1[0].OBJCAT['X_IMAGE'] - 1,
                               ad1[0].OBJCAT['Y_IMAGE'] - 1)
        ra2, dec2 = ad2[0].wcs(ad2[0].OBJCAT['X_IMAGE'] - 1,
                               ad2[0].OBJCAT['Y_IMAGE'] - 1)
        np.testing.assert_array_almost_equal(ra1, ra2)
        np.testing.assert_array_almost_equal(dec1, dec2)


@pytest.fixture(scope="function")
def gmos_tiled_images(astrofaker):
    """Create 4 GMOS images, already tiled into separate CCDs"""
    adinputs = []
    for i in (1, 2, 3, 4):
        ad = astrofaker.create('GMOS-N', ['IMAGE'],
                               filename=f"N20010101S{i:04d}.fits")
        ad.phu['DATALAB'] = f'GN-X-{i}'
        ad.init_default_extensions(overscan=False, binning=2)
        ad.add_read_noise()
        # To suppress the warning about mismatched gains
        ad.phu['GPREPARE'] = "YES"
        ad.hdr['GAIN'] = 1
        for ext in ad:
            ext.mask = np.zeros_like(ext.data, dtype=np.uint16)
        adinputs.append(ad)
    p = GMOSImage(adinputs)
    p.tileArrays(tile_all=False)
    return p.streams['main']
