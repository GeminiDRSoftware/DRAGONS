import pytest

import os

import astrodata, gemini_instruments
from astrodata.testing import ad_compare, download_from_archive

from recipe_system.reduction.coreReduce import Reduce


input_files = {"lampon": "S20240824S0067.fits",
               "lampoff": "S20240824S0068.fits",
               "dark": "S20240817S0018.fits"}


@pytest.mark.f2ls
@pytest.mark.parametrize("file1, file2, will_crash", [("lampon", "lampoff", False),
                                                      ("lampon", "dark", False),
                                                      ("lampoff", "dark", True)])
def test_make_processed_flat_partial(file1, file2, will_crash, path_to_refs, change_working_dir):
    """Check the makeProcessedFlat recipe ONLY as far as makeLampFlat"""
    files = [download_from_archive(input_files[f]) for f in [file1, file2]]

    crash = False
    with change_working_dir():
        r = Reduce()
        r.files.extend(files)
        r.uparms = [("makeLampFlat:write_outputs", True)]
        try:
            r.runr()
        except ValueError:
            crash = True

        assert crash == will_crash
        if not crash:
            adout = astrodata.open(r.output_filenames[0].replace("_flat", "_lampstack"))
            adref = astrodata.open(os.path.join(path_to_refs,
                                   adout.filename.replace("_lampstack", f"_{file1}_{file2}")))
            assert ad_compare(adout, adref, ignore="filename")
