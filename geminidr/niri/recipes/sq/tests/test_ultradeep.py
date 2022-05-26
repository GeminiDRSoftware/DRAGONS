import pytest
import os

from recipe_system.reduction.coreReduce import Reduce


niri_deep_dataset = [f"N20210512S{i:04d}.fits" for i in range(17, 35)]
niri_dark = "N20210512S0246_dark.fits"


@pytest.mark.slow
def test_ultradeep_recipe_niri(change_working_dir, path_to_inputs):
    """
    Test that the ultradeep recipe runs on NIRI images. This is not a
    regression test, it just confirms the absence of any errors.

    It also runs the makeSkyFlat recipe.
    """
    dark = os.path.join(path_to_inputs, niri_dark)
    with change_working_dir():
        skyflat = run_reduce(niri_deep_dataset, {"dark": dark},
                             recipe_name='makeSkyFlat')
        run_reduce(niri_deep_dataset, {"dark":dark,
                                       "flat": skyflat}, recipe_name='ultradeep')


def run_reduce(file_list, user_pars, recipe_name=None):
    r = Reduce()
    r.files = file_list
    r.uparms = user_pars
    if recipe_name:
        r.recipename = recipe_name

    r.runr()
    return r.output_filenames
