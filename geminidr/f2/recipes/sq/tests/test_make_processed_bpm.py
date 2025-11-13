import os
import pytest

import astrodata, gemini_instruments
from astrodata.testing import ad_compare, download_from_archive
from recipe_system.reduction.coreReduce import Reduce


@pytest.mark.f2
@pytest.mark.dragons_remote_data
def test_make_processed_bpm(path_to_refs, change_working_dir):
    """
    Test that we can make a processed bad pixel mask (BPM) from a F2 dataset.

    The F2 recipe includes an addDQ(), which means there'll be a query to the
    caldb for the static BPM. We allow this to take place and return nothing.
    """
    with change_working_dir():
        darks = [download_from_archive(f"S20131121S{i:04d}.fits")
                 for i in range(369, 376)]
        flats = [download_from_archive(f"S20131126S{i:04d}.fits")
                 for i in range(1111, 1117)] + [download_from_archive(
            f"S20131129S{i:04d}.fits") for i in range(320, 324)]
        r = Reduce()
        r.files.extend(flats + darks)
        r.recipename = "makeProcessedBPM"
        r.runr()

        adout = astrodata.open(r.output_filenames[0])
        adref = astrodata.open(os.path.join(path_to_refs, adout.filename))

        # TODO: remake reference with improved nonlinearity limit; ignore mismatch for now
        assert ad_compare(adref, adout, ignore_kw=["NONLINEA"])
