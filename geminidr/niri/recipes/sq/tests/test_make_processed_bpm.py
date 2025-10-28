import os
import pytest

import astrodata, gemini_instruments
from astrodata.testing import ad_compare, download_from_archive
from recipe_system.reduction.coreReduce import Reduce


@pytest.mark.niri
@pytest.mark.dragons_remote_data
def test_make_processed_bpm(path_to_refs, change_working_dir):
    """
    Test that we can make a processed bad pixel mask (BPM) from a NIRI dataset.
    """
    with change_working_dir():
        darks = [download_from_archive(f"N20160103S{i:04d}.fits")
                 for i in range(463, 473)]
        flats = [download_from_archive(f"N20160102S{i:04d}.fits")
                 for i in range(363, 383)]
        r = Reduce()
        r.files.extend(flats + darks)
        r.recipename = "makeProcessedBPM"
        r.runr()

        adout = astrodata.open(r.output_filenames[0])
        adref = astrodata.open(os.path.join(path_to_refs, adout.filename))

        assert ad_compare(adref, adout)
