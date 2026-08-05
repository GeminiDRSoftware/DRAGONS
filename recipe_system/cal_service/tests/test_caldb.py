# Various tests of the CalDB class and its subclasses
import pytest
import io
import os
from glob import glob

import astrodata, gemini_instruments

from recipe_system import cal_service
from recipe_system.config import globalConf

from geminidr.gmos.primitives_gmos_longslit import GMOSClassicLongslit

from recipe_system.cal_service.localmanager import LocalManagerError

CAL_DICT = {"N20180303S0131_nopixels.fits":
                {"processed_bias": "N20180302S0528_bias_nopixels.fits"}}


@pytest.fixture
def standard_config():
    """Populate globalConf object"""
    f = io.StringIO("[calibs]\ndatabases = ~/test/\n  fits\n")
    globalConf.read_file(f)


def cwd_config():
    """Populate globalConf object with a LocalDB in cwd"""
    f = io.StringIO(f"[calibs]\ndatabases = {os.getcwd()}/cal_manager.db\n")
    globalConf.read_file(f)


def test_local_database_configuration(standard_config):
    """Test the local database is read from the config file.
    This test will fail if there's a ~/.dragons/dragonsrc or
    ~/.geminidr/rsys.cfg file."""
    caldb = cal_service.set_local_database()
    assert len(caldb) == 1
    assert caldb.name == "~/test/cal_manager.db"
    assert caldb[0].name == "~/test/cal_manager.db"


def test_config_parsing(standard_config):
    p = GMOSClassicLongslit([])
    assert len(p.caldb) == 3
    assert p.caldb[1].name == "~/test/cal_manager.db"
    assert isinstance(p.caldb[1], cal_service.LocalDB)
    assert p.caldb[2].name == "fits"
    assert isinstance(p.caldb[2], cal_service.RemoteDB)


def test_localcaldb_init(tmp_path):
    """Test that caldb.init() for the local calibration manager works
    as expected"""

    # Configure DRAGONS to use database filename in the tmp_path
    dbfile = os.path.join(tmp_path, 'test.db')
    f = io.StringIO(f"[calibs]\ndatabases = {dbfile} get")
    globalConf.read_file(f)

    # Get a local caldb instance
    caldb = cal_service.set_local_database()

    # Initially, it should not exist
    assert os.path.exists(dbfile) is False

    # Init the DB, check that it now exists and note the mtime for later
    caldb.init()
    assert os.path.exists(dbfile) is True
    mtime = os.path.getmtime(dbfile)

    # Init it again, we should get an error
    with pytest.raises(LocalManagerError):
        caldb.init()

    # Init it again with wipe=True and check new file exists
    caldb.init(wipe=True)
    assert os.path.exists(dbfile) is True
    assert os.path.getmtime(dbfile) > mtime


@pytest.mark.preprocessed_data
def test_api_store_and_delete(path_to_inputs, change_working_dir):
    with change_working_dir():
        cwd_config()
        caldb = cal_service.set_local_database()
        caldb.init()

        # Any file will do, take the first
        cal_file = sorted(glob(os.path.join(path_to_inputs,
                                            "*bias*.fits")))[0]
        cal_name = os.path.basename(cal_file)
        caldb.add_cal(cal_file)
        assert len(list(caldb.list_files())) == 1
        assert list(caldb.list_files())[0].name == cal_name

        # Now remove it
        caldb.remove_cal(cal_name)
        assert len(list(caldb.list_files())) == 0

        # Clean up
        os.remove(caldb.name)


@pytest.mark.preprocessed_data
def test_retrieval(path_to_inputs, change_working_dir):
    with change_working_dir():
        cwd_config()
        caldb = cal_service.set_local_database()

        for sci, cals in CAL_DICT.items():
            caldb.init()
            for cal in cals.values():
                cal_file = os.path.join(path_to_inputs, cal)
                caldb.add_cal(cal_file)

            ad_sci = astrodata.open(os.path.join(path_to_inputs, sci))
            for caltype, calfile in cals.items():
                cal_return = caldb.get_calibrations([ad_sci], caltype=caltype)
                assert len(cal_return) == 1
                assert cal_return.files[0] == cal_file
                assert cal_return.origins[0] == caldb.name

            # Delete caldb
            os.remove(caldb.name)
