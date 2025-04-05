import pytest

import astrodata, gemini_instruments
from geminidr.gmos.primitives_gmos import GMOS
from geminidr.gmos.recipes.sq.recipes_BIAS import makeProcessedBias

from astrodata.testing import download_from_archive


@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("primitive_name,num_outputs",
                         [("stackFrames", 1), ("stackBiases", 3)])
def test_skip_primitive(change_working_dir, primitive_name, num_outputs):
    """Reduce some biases and confirm that the correct thing is skipped (or not)"""
    with change_working_dir():
        files = [download_from_archive(f"N20210101S{i:04d}.fits") for i in range(534, 537)]
        adinputs = [astrodata.from_file(f) for f in files]
        p = GMOS(adinputs, uparms={f'{primitive_name}:skip_primitive': True})
        makeProcessedBias(p)
        assert len(p.streams['main']) == num_outputs
