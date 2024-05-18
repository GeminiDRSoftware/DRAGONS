import pytest

from geminidr.ghost.primitives_ghost_slit import GHOSTSlit

from . import ad_slit


@pytest.mark.ghostslit
@pytest.mark.skip(reason='Needs to be tested with a reduced slit flat - '
                         'full reduction test required')
def test_weightSlitExposures(ad_slit):
    """
    Checks to make:

    - Ensure the slit viewer bundle ends up with:

        a) A mean exposure epoch - DONE in test_slitarc_procslit_done
        b) The correct mean exposure epoch - DONE in test_slitarc_avgepoch
    """
    p = GHOSTSlit([ad_slit])
    p.weightSlitExposures()
    assert ad_slit.phu.get('AVGEPOCH') is not None
