import pytest
from copy import deepcopy

import numpy as np
from astropy.table import Table, vstack
from astropy import units as u
from astropy.coordinates import FK5
from astropy.modeling import models
from gwcs import coordinate_frames as cf
from gwcs.wcs import WCS as gWCS

from astrodata import wcs as adwcs
from gempy.library import astromodels as am
from geminidr.niri.primitives_niri_image import NIRIImage
from geminidr.gmos.primitives_gmos_longslit import GMOSLongslit


@pytest.mark.parametrize("tolerance", (0, 1))
@pytest.mark.parametrize("use_common", (False, True))
def test_scale_by_object_flux_image(p_niri_image, tolerance, use_common):
    expected_results = {(0, False): (1, 1/2, 1/3),
                        (0, True): (1, 1/2, 1/3),
                        (1, False): (1, 170100/170201, 100200/100404),  # 0.9994054, 0.00796915
                        (1, True): (1, 100/101, 100/102)  # 0.990985, 0.9803924
                        }

    p_niri_image.scaleByObjectFlux(tolerance=tolerance, use_common=use_common)
    adoutputs = p_niri_image.streams['main']
    scalings = [ad[0].data.mean() for ad in adoutputs]
    np.testing.assert_allclose(scalings, expected_results[tolerance, use_common],
                               rtol=1e-5)


@pytest.mark.parametrize("tolerance", (0, 1))
@pytest.mark.parametrize("use_common", (False, True))
@pytest.mark.parametrize("extracted", (False, True))
def test_scale_by_object_flux_spect(p_gmos_ls, tolerance, use_common, extracted):
    expected_results = {(0, False): (1, 1/2, 1/3),
                        (0, True): (1, 1/2, 1/3),
                        (1, False): (1, 360/371, 745/769),
                        (1, True): (1, 335/346, 345/369)
                        }

    if extracted:
        p_gmos_ls.extractSpectra()
    adinputs = [deepcopy(ad) for ad in p_gmos_ls.streams['main']]
    p_gmos_ls.scaleByObjectFlux(tolerance=tolerance, use_common=use_common)
    adoutputs = p_gmos_ls.streams['main']
    # Only need to look at first extension, even for multi-extension 1D spectra
    scalings = [np.mean(adout[0].data[adout[0].data > 0] /
                        adin[0].data[adin[0].data > 0])
                for adin, adout in zip(adinputs, adoutputs)]
    np.testing.assert_allclose(scalings, expected_results[tolerance, use_common],
                               rtol=1e-5)


@pytest.fixture
def p_niri_image(astrofaker):
    full_objcat = Table([[100.0, 100.1, 100.2, 100.3],
                         [5.0, 5.1, 5.2, 5.3],
                         [100., 200., 300., 400.],
                         [2., 2., 2., 2.]],
                        names=('X_WORLD', 'Y_WORLD', 'FLUX_AUTO', 'FLUXERR_AUTO'))
    adinputs = []
    for i in range(1, 4):
        ad = astrofaker.create('NIRI', 'IMAGE', filename=f'N20010101S{i:04d}.fits')
        ad.init_default_extensions()
        ad.add(1)
        ad.exposure_time = i * 10
        rows = np.ones(len(full_objcat), dtype=bool)
        rows[i] = False
        ad[0].OBJCAT = full_objcat[rows]
        ad[0].OBJCAT['FLUX_AUTO'][0] += i - 1
        adinputs.append(ad)

    return NIRIImage(adinputs)


@pytest.fixture
def p_gmos_ls():
    """A pretty messy fixture to create 3 fake 2D spectrograms"""
    import astrofaker

    # Create a longslit gWCS object
    m_wave = models.Shift(500, name='WAVE')
    m_sky = (models.Mapping((0, 0)) | (models.Shift(-125) & models.Shift(0)) |
             models.AffineTransformation2D([[-0.00004, 0], [0, 0.00004]],
                                           name='cd_matrix') |
             models.Pix2Sky_Gnomonic() |
             models.RotateNative2Celestial(lon=30, lat=20, lon_pole=180))
    input_frame = adwcs.pixel_frame(2)
    output_frame = cf.CompositeFrame(
        [cf.SpectralFrame(name="Wavelength in air", unit=u.nm,
                          axes_names=('AWAV',), axes_order=(0,)),
         cf.CelestialFrame(reference_frame=FK5(), name='SKY',
                           axes_names=('lon', 'lat'), axes_order=(1, 2))])

    data = np.zeros((250, 200))
    for i in range(50, 201, 50):
        data[i] = i

    adinputs = []
    for i in range(1, 4):
        shift = (0, -60, 60)[i - 1]
        ad = astrofaker.create('GMOS-N', 'SPECT',
                               filename=f'N20010101S{i:04d}.fits',
                               extra_keywords={'SKYCORR': 'YES'})
        ad.add_extension(shape=data.shape, pixel_scale=0.1)
        ad[0].wcs = gWCS([(input_frame, m_wave & (models.Shift(shift) | m_sky)),
                          (output_frame, None)])
        y1 = max(-shift, 0)
        y2 = min(-shift + data.shape[0], data.shape[0])
        iy1 = max(shift, 0)
        ad[0].data[y1:y2] = data[iy1:iy1 + y2 - y1]
        aptables = []
        for row in np.where(ad[0].data[:, 0] > 0)[0]:
            m = models.Chebyshev1D(degree=0, c0=row, domain=[0, data.shape[1] - 1])
            aptables.append(am.model_to_table(m))
        aptable = vstack(aptables, metadata_conflicts="silent")
        aptable['number'] = np.arange(len(aptable)) + 1
        aptable['aper_lower'] = -2
        aptable['aper_upper'] = 2
        new_col_order = (["number"] + sorted(c for c in aptable.colnames
                                             if c.startswith("c")) +
                         ["aper_lower", "aper_upper"])
        ad[0].APERTURE = aptable[new_col_order]
        ad.phu[ad._keyword_for('exposure_time')] = i * 10
        adinputs.append(ad)
        data[100] += 10

    return GMOSLongslit(adinputs)
