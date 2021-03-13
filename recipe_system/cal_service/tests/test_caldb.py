# Various tests of the CalDB class and its subclasses
import pytest
import io
import os

from recipe_system import cal_service
from recipe_system.config import globalConf

from geminidr.gmos.primitives_gmos_longslit import GMOSLongslit

@pytest.fixture
def standard_config():
    """Populate globalConf object"""
    f = io.StringIO("[calibs]\ndatabases = ~/test/\n  fits\n")
    globalConf.read_file(f)


def cwd_config():
    """Populate globalConf object with a LocalDB in cwd"""
    f = io.StringIO(f"[calibs]\ndatabases = {os.getcwd()}/\n")
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
    p = GMOSLongslit([])
    assert len(p.caldb) == 3
    assert p.caldb[1].name == "~/test/cal_manager.db"
    assert isinstance(p.caldb[1], cal_service.LocalDB)
    assert p.caldb[2].name == "fits"
    assert isinstance(p.caldb[2], cal_service.RemoteDB)


def test_api_store(path_to_inputs, change_working_dir):
    with change_working_dir():
        cwd_config()
        caldb = cal_service.set_local_database()
        caldb.init()
        bias_file = os.path.join(path_to_inputs,
                                 "N20201022S0160_bias_nopixels.fits")
        caldb.add_cal(bias_file)
        assert len(list(caldb.list_files())) == 1
        assert list(caldb.list_files())[0].name == os.path.basename(bias_file)


def test_retrieval(path_to_inputs, change_working_dir):
    pass
