# A full end-to-end reduction of some GHOST data
import os
import pytest

from astrodata.testing import download_from_archive
from gempy.utils import logutils
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.cal_service import LocalDB, get_db_path_from_config
from recipe_system.config import load_config

CONFIG_FILE = "dragonsrc"


datasets = {
    "GS-2023B-FT-105-2-081": {
        "science": [f"S20231216S{i:04d}.fits" for i in range(38, 41)],
        "bias": ["S20231217S0093.fits"],
        "flat": ["S20231217S0123.fits"],
        "arc": ["S20231217S0121.fits"],
        "calbias": ["S20231217S0099.fits"],
        "standard": ["S20231211S0287.fits"],
        "stdbias": ["S20231212S0003.fits"],
        "stdflat": ["S20231212S0011.fits"],
        "scibias": ["S20231217S0093.fits"],
    }
}


def initialize_database(path_to_inputs, filename=CONFIG_FILE):
    cwd = os.getcwd()
    with open(filename, "w") as f:
        f.write(f"[calibs]\ndatabases = {cwd}/dragons.db get\n")
    try:
        os.remove("dragons.db")
    except FileNotFoundError:
        pass

    load_config(filename)
    db_path = get_db_path_from_config()
    caldb = LocalDB(db_path, force_init=True)

    caldb.store_calibration(os.path.join(path_to_inputs, "bpm_20220601_ghost_blue_11_full_4amp.fits"), caltype="processed_bpm")
    caldb.store_calibration(os.path.join(path_to_inputs, "bpm_20220601_ghost_red_11_full_4amp.fits"), caltype="processed_bpm")
    return caldb


@pytest.mark.slow
@pytest.mark.integration_test
@pytest.mark.ghost_integ
@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("progid,file_dict", [[k] + [v] for k, v in list(datasets.items())])
def test_reduce_ghost(change_working_dir, path_to_inputs, progid, file_dict):

    with change_working_dir(progid):
        caldb = initialize_database(path_to_inputs)

        # BIAS for calibrations
        calbias_files = file_dict["calbias"]
        calbias_paths = [download_from_archive(f) for f in calbias_files]
        debundle_outputs = reduce(calbias_paths, f"calbias_bundle_{progid}", caldb)
        for arm in ("slit", "blue", "red"):
            calbias_files = [f for f in debundle_outputs if arm in f]
            reduce(calbias_files, f"calbias_{arm}_{progid}", caldb,
                   caltype="processed_bias")

        # FLAT
        flat_files = file_dict["flat"]
        flat_paths = [download_from_archive(f) for f in flat_files]
        debundle_outputs = reduce(flat_paths, f"flat_bundle_{progid}", caldb)
        for arm in ("slit", "blue", "red"):
            flat_files = [f for f in debundle_outputs if arm in f]
            user_pars = None if arm == "slit" else [("traceFibers:smoothing", 6)]
            reduce(flat_files, f"flat_{arm}_{progid}", caldb,
                   caltype="processed_slitflat" if arm == "slit"
                   else "processed_flat", user_pars=user_pars)

        # ARC
        arc_files = file_dict["arc"]
        arc_paths = [download_from_archive(f) for f in arc_files]
        debundle_outputs = reduce(arc_paths, f"arc_bundle_{progid}", caldb)
        for arm in ("slit", "blue", "red"):
            arc_files = [f for f in debundle_outputs if arm in f]
            reduce(arc_files, f"arc_{arm}_{progid}", caldb,
                   caltype="processed_slit" if arm == "slit"
                   else "processed_arc")

        # BIAS for standard: don't need the slit bias
        try:
            stdbias_files = file_dict["stdbias"]
        except KeyError:
            pass
        else:
            stdbias_paths = [download_from_archive(f) for f in stdbias_files]
            debundle_outputs = reduce(stdbias_paths, f"stdbias_bundle_{progid}", caldb)
            for arm in ("blue", "red"):
                stdbias_files = [f for f in debundle_outputs if arm in f]
                reduce(stdbias_files, f"stdbias_{arm}_{progid}", caldb,
                       caltype="processed_bias")

        # FLAT for standard
        try:
            stdflat_files = file_dict["stdflat"]
        except KeyError:
            pass
        else:
            stdflat_paths = [download_from_archive(f) for f in stdflat_files]
            debundle_outputs = reduce(stdflat_paths, f"stdflat_bundle_{progid}",
                                      caldb)
            for arm in ("slit", "blue", "red"):
                stdflat_files = [f for f in debundle_outputs if arm in f]
                user_pars = None if arm == "slit" else [("traceFibers:smoothing", 6)]
                reduce(stdflat_files, f"stdflat_{arm}_{progid}", caldb,
                       caltype="processed_slitflat" if arm == "slit"
                       else "processed_flat", user_pars=user_pars)

        # STANDARD
        stdbias_files = file_dict["standard"]
        stdbias_paths = [download_from_archive(f) for f in stdbias_files]
        debundle_outputs = reduce(stdbias_paths, f"standard_bundle_{progid}", caldb)
        for arm in ("slit", "blue", "red"):
            standard_files = [f for f in debundle_outputs if arm in f]
            kwargs = ({"caltype": "processed_slit"} if arm == "slit" else
                      {"caltype": "processed_standard",
                       "recipe_name": "reduceStandard",
                       "user_pars": [("calculateSensitivity:filename",
                                      os.path.join(path_to_inputs,
                                                   "gd71_stiswfcnic_004.fits"))]})
            reduce(standard_files, f"stdbias_{arm}_{progid}", caldb, **kwargs)

    # BIAS for science: don't need the slit bias
    try:
        scibias_files = file_dict["scibias"]
    except KeyError:
        pass
    else:
        scibias_paths = [download_from_archive(f) for f in scibias_files]
        debundle_outputs = reduce(scibias_paths, f"scibias_bundle_{progid}", caldb)
        for arm in ("blue", "red"):
            scibias_files = [f for f in debundle_outputs if arm in f]
            reduce(scibias_files, f"scibias_{arm}_{progid}", caldb,
                   caltype="processed_bias")

    # SCIENCE: proprietary so can't download from archive
    science_files = file_dict["science"]
    science_paths = [os.path.join(path_to_inputs, f) for f in science_files]
    debundle_outputs = reduce(science_paths, f"science_bundle_{progid}", caldb)
    for arm in ("slit", "blue", "red"):
        science_files = [f for f in debundle_outputs if arm in f]
        kwargs = {"caltype": "processed_slit"} if arm == "slit" else {}
        user_pars = [("fluxCalibrate:do_cal", "skip")] if arm != "slit" else []
        reduce(science_files, f"science_{arm}_{progid}", caldb,
               user_pars=user_pars, **kwargs)


# -- Helper functions ---------------------------------------------------------
def reduce(file_list, label, caldb, recipe_name=None,
           caltype=None, user_pars=None):
    """
    Helper function used to prevent replication of code.

    Parameters
    ----------
    file_list : list
        List of files that will be reduced.
    label : str
        Labed used on log files name.
    all_cals : list
        List of all calibration files properly formatted for DRAGONS Reduce().
    cals : list/None
        List of calibrations to use for this particular reduction
    recipe_name : str, optional
        Name of the recipe used to reduce the data.
    save_to : str, optional
        Stores the calibration files locally in a list.
    user_pars : list, optional
        List of user parameters

    Returns
    -------
    str : Output reduced filenames
    list : An updated list of calibration files.
    """
    objgraph = pytest.importorskip("objgraph")

    logutils.get_logger().info("\n\n\n")
    logutils.config(file_name=f"test_image_{label}.log")
    r = Reduce()
    r.files = file_list
    r.uparms = user_pars
    r.config_file = os.path.join(os.getcwd(), CONFIG_FILE)

    if recipe_name:
        r.recipename = recipe_name

    r.runr()

    if caltype:
        # Add calibrations manually rather than auto-store because of
        # possible cross-talk with other tests
        for f in r.output_filenames:
            caldb.store_calibration(f"{os.path.join('calibrations', caltype, f)}", caltype=caltype)
        [os.remove(f) for f in r.output_filenames]

    # check that we are not leaking objects
    assert len(objgraph.by_type('NDAstroData')) == 0

    return r.output_filenames