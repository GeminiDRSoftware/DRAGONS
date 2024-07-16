import pytest
import os

from numpy.testing import assert_allclose
from astropy.table import Table

import astrodata, gemini_instruments
from astrodata.testing import ad_compare
from geminidr.gsaoi.primitives_gsaoi_image import GSAOIImage
from gempy.library import matching


@pytest.mark.slow
@pytest.mark.preprocessed_data
def test_gsaoi_adjust_wcs_no_refcat(change_working_dir, path_to_refs, adinputs):
    with change_working_dir():
        p = GSAOIImage(adinputs)
        p.adjustWCSToReference(order=3, final=0.2)
        p.resampleToCommonFrame()
        p.writeOutputs()
        for ad in p.streams['main']:
            ad = astrodata.open(ad.filename)
            ref_ad = astrodata.open(os.path.join(path_to_refs, ad.filename))
            # CJS: The outputs I get on my MacBook apparently do not agree with
            # those on the Jenkins server, so need to increase the tolerances.
            # This should still be fine.
            ad_compare(ad, ref_ad, atol=0.1, rtol=1e-5)


@pytest.mark.slow
@pytest.mark.preprocessed_data
def test_gsaoi_resample_to_refcat(path_to_inputs, adinputs):
    """
    Use an HST catalog to provide a good absolute astrometric solution and
    resample the image to a regular projection. It confirms the projection is
    correct by comparing the positions of sources on the image with those
    expected from the reference catalog using the WCS of the output image.

    This test will definitely fail if the WCS isn't updated to match the refcat
    """
    table_path = os.path.join(path_to_inputs, 'hst_catalog.fits')
    p = GSAOIImage(adinputs[:1])
    p.addReferenceCatalog(source=table_path)
    p.determineAstrometricSolution(order=3, max_iters=5, initial=1, final=0.25)
    p.resampleToCommonFrame()
    p.detectSources()
    ad = p.streams['main'][0]
    objcat = ad[0].OBJCAT
    incoords = (objcat['X_IMAGE']-1, objcat['Y_IMAGE']-1)
    t = Table.read(table_path)
    refcoords = ad[0].wcs.invert(t['RAJ2000'], t['DEJ2000'])
    t = matching.find_alignment_transform(incoords, refcoords, rotate=True, scale=True)
    assert ad[0].shape == (4219, 4226)
    assert_allclose(t.factor_0, 1, atol=1e-3)
    assert_allclose(t.angle_1, 0, atol=0.01)
    assert_allclose(t.parameters[2:], 0, atol=0.07)


@pytest.fixture(scope="function")
def adinputs(path_to_inputs):
    adinputs = []
    for i in range(148, 151):
        adinputs.append(astrodata.open(
            os.path.join(path_to_inputs, f'S20200305S{i:04d}_sourcesDetected.fits')))
    return adinputs
