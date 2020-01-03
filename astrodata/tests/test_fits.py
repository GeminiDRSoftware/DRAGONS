import os

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.modeling import models

import astrodata
from astrodata.testing import download_from_archive, compare_models

test_files = [
    "N20160727S0077.fits",  # NIFS DARK
    "N20170529S0168.fits",  # GMOS-N SPECT
    "N20190116G0054i.fits",  # GRACES SPECT
    "N20190120S0287.fits",  # NIRI IMAGE
    "N20190206S0279.fits",  # GNIRS SPECT XD
    "S20150609S0023.fits",  # GSAOI DARK
    "S20170103S0032.fits",  # F2 IMAGE
    "S20170505S0031.fits",  # GSAOI FLAT
    "S20170505S0095.fits",  # GSAOI IMAGE
    "S20171116S0078.fits",  # GMOS-S MOS NS
    "S20180223S0229.fits",  # GMOS IFU ACQUISITION
    "S20190213S0084.fits",  # F2 IMAGE
]


@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_ad_basics(input_file):
    fname = download_from_archive(input_file)
    ad = astrodata.open(fname)

    assert isinstance(ad, astrodata.fits.AstroDataFits)
    assert ad.filename == os.path.basename(fname)
    assert type(ad[0].data) == np.ndarray

    metadata = (('SCI', 1), ('SCI', 2), ('SCI', 3))

    for ext, md in zip(ad, metadata):
        assert ext.hdr['EXTNAME'] == md[0]
        assert ext.hdr['EXTVER'] == md[1]


@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_can_add_and_del_extension(input_file):
    fname = download_from_archive(input_file)
    ad = astrodata.open(fname)
    original_size = len(ad)

    ourarray = np.array([(1, 2, 3), (11, 12, 13), (21, 22, 23)])
    ad.append(ourarray)
    assert len(ad) == original_size + 1

    del ad[original_size]
    assert len(ad) == original_size


@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_slice(input_file):
    fname = download_from_archive(input_file)
    ad = astrodata.open(fname)
    assert ad.is_sliced is False

    # single
    try:
        metadata = ('SCI', 2)
        ext = ad[1]
    except IndexError:
        assert len(ad) == 1  # some files does not have enough extensions
    else:
        assert ext.is_single is True
        assert ext.is_sliced is True
        assert ext.hdr['EXTNAME'] == metadata[0]
        assert ext.hdr['EXTVER'] == metadata[1]

    # multiple
    metadata = ('SCI', 2), ('SCI', 3)
    try:
        slc = ad[1, 2]
    except IndexError:
        assert len(ad) == 1  # some files does not have enough extensions
    else:
        assert len(slc) == 2
        assert ext.is_sliced is True
        for ext, md in zip(slc, metadata):
            assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == md

    # iterate over single slice
    metadata = ('SCI', 1)
    for ext in ad[0]:
        assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == metadata

    # slice negative
    assert ad.data[-1] is ad[-1].data


@pytest.mark.dragons_remote_data
def test_phu():
    fname = download_from_archive(test_files[0])
    ad = astrodata.open(fname)

    # The result of this depends if gemini_instruments was imported or not
    # assert ad.descriptors == ('instrument', 'object', 'telescope')
    # assert ad.tags == set()

    assert ad.instrument() == 'NIFS'
    assert ad.object() == 'Dark'
    assert ad.telescope() == 'Gemini-North'

    ad.phu['DETECTOR'] = 'FooBar'
    ad.phu['ARBTRARY'] = 'BarBaz'

    assert ad.phu['DETECTOR'] == 'FooBar'
    assert ad.phu['ARBTRARY'] == 'BarBaz'

    if ad.instrument().upper() not in ['GNIRS', 'NIRI', 'F2']:
        del ad.phu['DETECTOR']
        assert 'DETECTOR' not in ad.phu


@pytest.mark.dragons_remote_data
def test_writes_to_new_fits(tmpdir):
    fname = download_from_archive(test_files[0])
    ad = astrodata.open(fname)

    testfile = str(tmpdir.join('temp.fits'))
    ad.write(testfile)
    assert os.path.exists(testfile)

    # overwriting is forbidden by default
    with pytest.raises(OSError):
        ad.write(testfile)

    ad.write(overwrite=True)
    assert os.path.exists(testfile)


def test_can_make_and_write_ad_object(tmpdir):
    # Creates data and ad object
    phu = fits.PrimaryHDU()
    pixel_data = np.random.rand(100, 100)

    hdu = fits.ImageHDU()
    hdu.data = pixel_data

    ad = astrodata.create(phu)
    ad.append(hdu, name='SCI')

    # Write file and test it exists properly
    testfile = str(tmpdir.join('created_fits_file.fits'))
    ad.write(testfile)

    # Opens file again and tests data is same as above
    adnew = astrodata.open(testfile)
    assert np.array_equal(adnew[0].data, pixel_data)


def test_can_append_table_and_access_data(capsys, tmpdir):
    tbl = Table([np.zeros(10), np.ones(10)], names=['col1', 'col2'])
    phu = fits.PrimaryHDU()
    ad = astrodata.create(phu)
    astrodata.add_header_to_table(tbl)
    ad.append(tbl, name='BOB')
    assert ad.exposed == {'BOB'}

    ad.info()
    captured = capsys.readouterr()
    assert '.BOB           Table       (10, 2)' in captured.out

    # Write file and test it exists properly
    testfile = str(tmpdir.join('created_fits_file.fits'))
    ad.write(testfile)
    adnew = astrodata.open(testfile)
    assert adnew.exposed == {'BOB'}
    assert len(adnew.BOB) == 10


@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_set_a_keyword_on_phu_deprecated(input_file):
    fname = download_from_archive(input_file)
    ad = astrodata.open(fname)

    try:
        with pytest.raises(AssertionError):
            ad.phu.DETECTOR = 'FooBar'
            ad.phu.ARBTRARY = 'BarBaz'

            assert ad.phu.DETECTOR == 'FooBar'
            assert ad.phu.ARBTRARY == 'BarBaz'
            assert ad.phu['DETECTOR'] == 'FooBar'
    except KeyError as e:
        # Some instruments don't have DETECTOR as a keyword
        if e.args[0] == "Keyword 'DETECTOR' not found.":
            pass
        else:
            raise KeyError


# Regression:
# Make sure that references to associated
# extension objects are copied across
@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_do_arith_and_retain_features(input_file):
    fname = download_from_archive(input_file)
    ad = astrodata.open(fname)

    ad[0].NEW_FEATURE = np.array([1, 2, 3, 4, 5])
    ad2 = ad * 5
    np.testing.assert_array_almost_equal(ad[0].NEW_FEATURE, ad2[0].NEW_FEATURE)


def test_update_filename():
    phu = fits.PrimaryHDU()
    ad = astrodata.create(phu)
    ad.filename = 'myfile.fits'

    # This will also set ORIGNAME='myfile.fits'
    ad.update_filename(suffix='_suffix1')
    assert ad.filename == 'myfile_suffix1.fits'

    ad.update_filename(suffix='_suffix2', strip=True)
    assert ad.filename == 'myfile_suffix2.fits'

    ad.update_filename(suffix='_suffix1', strip=False)
    assert ad.filename == 'myfile_suffix2_suffix1.fits'

    ad.filename = 'myfile.fits'
    ad.update_filename(prefix='prefix_', strip=True)
    assert ad.filename == 'prefix_myfile.fits'

    ad.update_filename(suffix='_suffix', strip=True)
    assert ad.filename == 'prefix_myfile_suffix.fits'

    ad.update_filename(prefix='', suffix='_suffix2', strip=True)
    assert ad.filename == 'myfile_suffix2.fits'

    # Now check that updates are based on existing filename
    # (so "myfile" shouldn't appear)
    ad.filename = 'file_suffix1.fits'
    ad.update_filename(suffix='_suffix2')
    assert ad.filename == 'file_suffix1_suffix2.fits'

    # A suffix shouldn't have an underscore, so should assume that
    # "file_suffix1" is the root
    ad.update_filename(suffix='_suffix3', strip=True)
    assert ad.filename == 'file_suffix1_suffix3.fits'


@pytest.mark.dragons_remote_data
def test_read_a_keyword_from_phu_deprecated():
    "Test deprecated methods to access headers"
    ad = astrodata.open(
        download_from_archive('N20110826S0336.fits', path='GMOS'))

    with pytest.raises(AttributeError):
        assert ad.phu.DETECTOR == 'GMOS + Red1'

    with pytest.raises(AttributeError):
        assert ad.hdr.CCDNAME == [
            'EEV 9273-16-03', 'EEV 9273-20-04', 'EEV 9273-20-03'
        ]

    # and when accessing missing extension
    with pytest.raises(AttributeError):
        ad.ABC


def test_round_trip_gwcs():
    """
    Add a 2-step gWCS instance to NDAstroData, save to disk, reload & compare.
    """

    from gwcs import coordinate_frames as cf
    from gwcs import WCS

    arr = np.zeros((10, 10), dtype=np.float32)
    ad1 = astrodata.create(fits.PrimaryHDU(), [fits.ImageHDU(arr, name='SCI')])

    # Transformation from detector pixels to pixels in some reference row,
    # removing relative distortions in wavelength:
    det_frame = cf.Frame2D(name='det_mosaic', axes_names=('x', 'y'),
                           unit=(u.pix, u.pix))
    dref_frame = cf.Frame2D(name='dist_ref_row', axes_names=('xref', 'y'),
                            unit=(u.pix, u.pix))

    # A made-up example model that looks vaguely like some real distortions:
    fdist = models.Chebyshev2D(2, 2,
                               c0_0=4.81125, c1_0=5.43375, c0_1=-0.135,
                               c1_1=-0.405, c0_2=0.30375, c1_2=0.91125,
                               x_domain=(0., 9.), y_domain=(0., 9.))

    # This is not an accurate inverse, but will do for this test:
    idist = models.Chebyshev2D(2, 2,
                               c0_0=4.89062675, c1_0=5.68581232,
                               c2_0=-0.00590263, c0_1=0.11755526,
                               c1_1=0.35652358, c2_1=-0.01193828,
                               c0_2=-0.29996306, c1_2=-0.91823397,
                               c2_2=0.02390594,
                               x_domain=(-1.5, 12.), y_domain=(0., 9.))

    # The resulting 2D co-ordinate mapping from detector to ref row pixels:
    distrans = models.Mapping((0, 1, 1)) | (fdist & models.Identity(1))
    distrans.inverse = models.Mapping((0, 1, 1)) | (idist & models.Identity(1))

    # Transformation from reference row pixels to linear, row-stacked spectra:
    spec_frame = cf.SpectralFrame(axes_order=(0,), unit=u.nm,
                                  axes_names='lambda', name='wavelength')
    row_frame = cf.CoordinateFrame(1, 'SPATIAL', axes_order=(1,), unit=u.pix,
                                   axes_names='y', name='row')
    rss_frame = cf.CompositeFrame([spec_frame, row_frame])

    # Toy wavelength model & approximate inverse:
    fwcal = models.Chebyshev1D(2, c0=500.075, c1=0.05, c2=0.001, domain=(0, 9))
    iwcal = models.Chebyshev1D(2, c0=4.59006292, c1=4.49601817, c2=-0.08989608,
                               domain=(500.026, 500.126))

    # The resulting 2D co-ordinate mapping from ref pixels to wavelength:
    wavtrans = fwcal & models.Identity(1)
    wavtrans.inverse = iwcal & models.Identity(1)

    # The complete WCS chain for these 2 transformation steps:
    ad1[0].nddata.wcs = WCS([(det_frame, distrans),
                             (dref_frame, wavtrans),
                             (rss_frame, None)
                            ])

    # Save & re-load the AstroData instance with its new WCS attribute:
    ad1.write('round_trip_gwcs.fits')
    ad2 = astrodata.open('round_trip_gwcs.fits')

    wcs1 = ad1[0].nddata.wcs
    wcs2 = ad2[0].nddata.wcs

    # # Temporary workaround for issue #9809, to ensure the test is correct:
    # wcs2.forward_transform[1].x_domain = (0, 9)
    # wcs2.forward_transform[1].y_domain = (0, 9)
    # wcs2.forward_transform[3].domain = (0, 9)
    # wcs2.backward_transform[0].domain = (500.026, 500.126)
    # wcs2.backward_transform[3].x_domain = (-1.5, 12.)
    # wcs2.backward_transform[3].y_domain = (0, 9)

    # Did we actually get a gWCS instance back?
    assert isinstance(wcs2, WCS)

    # Do the transforms have the same number of submodels, with the same types,
    # degrees, domains & parameters? Here the inverse gets checked redundantly
    # as both backward_transform and forward_transform.inverse, but it would be
    # convoluted to ensure that both are correct otherwise (since the transforms
    # get regenerated as new compound models each time they are accessed).
    compare_models(wcs1.forward_transform, wcs2.forward_transform)
    compare_models(wcs1.backward_transform, wcs2.backward_transform)

    # Also compare a few transformed values, as the "proof of the pudding":
    y, x = np.mgrid[0:9:2, 0:9:2]
    np.testing.assert_allclose(wcs1(x, y), wcs2(x, y), rtol=1e-7, atol=0.)

    y, w = np.mgrid[0:9:2, 500.025:500.12:0.0225]
    np.testing.assert_allclose(wcs1.invert(w, y), wcs2.invert(w, y),
                               rtol=1e-7, atol=0.)


if __name__ == '__main__':
    pytest.main()
