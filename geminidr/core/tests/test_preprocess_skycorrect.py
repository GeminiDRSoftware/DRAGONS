import pytest

from astropy.table import Table
from geminidr.niri.primitives_niri_image import NIRIImage
from gempy.gemini import gemini_tools as gt


def test_skycorrect_twoimages(astrofaker):
    """
    Confirm that we can do pairwise sky subtraction (A-B, B-A) correctly.
    A has a sky level of 1000, B 2000; both have an object with max=1000.
    So, after sky-subtraction, A's max should be 500 (A-0.5*B) and B's min
    should be -1000 (B-2*A).
    """
    adinputs = []
    for i in (1, 2):
        ad = astrofaker.create("NIRI", ["IMAGE"], filename=f"ad{i}.fits")
        ad.init_default_extensions()
        ad.add_read_noise()
        ad[0].add(i * 1000)
        ad[0].add_star(amplitude=1000, x=512, y=512)
        adinputs.append(ad)

    for i, ad in enumerate(adinputs):
        skytable = Table([[adinputs[1-i].filename]], names=["SKYNAME"])
        ad.SKYTABLE = skytable

    p = NIRIImage(adinputs)
    p.skyCorrect(mask_objects=False)
    # Large tolerances to deal with unlikely random numbers
    assert p.streams['main'][0][0].data.max() == pytest.approx(500, abs=50)
    assert p.streams['main'][1][0].data.min() == pytest.approx(-1000, abs=50)
    for ad in p.streams['main']:
        assert gt.measure_bg_from_image(ad[0], value_only=True) == pytest.approx(0, abs=5)
