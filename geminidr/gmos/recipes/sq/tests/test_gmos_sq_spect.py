#!/usr/bin/python
"""
Tests related to GMOS Long-slit Spectroscopy data reduction.
"""
import glob
import os
import shutil

import astrodata
import geminidr

# noinspection PyPackageRequirements
import pytest

# noinspection PyUnresolvedReferences
import gemini_instruments

from geminidr.gmos import primitives_gmos_spect, primitives_gmos_longslit
from gempy.adlibrary import dataselect
from gempy.utils import logutils
from recipe_system import cal_service
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals

dataset_folder_list = [
    'GMOS/GN-2017A-FT-19',
    # 'GMOS/GS-2016B-Q-54-32'
]


@pytest.fixture(scope='class', params=dataset_folder_list)
def config(request, path_to_inputs, path_to_outputs):
    """
    Super fixture that returns an object with the data required for the tests
    inside this file. This super fixture avoid confusions with Pytest, Fixtures
    and Parameters that could generate a very large matrix of configurations.

    The `path_to_*` fixtures are defined inside the `conftest.py` file.

    Parameters
    ----------
    request : pytest.fixture
        A special fixture providing information of the requesting test function.
    path_to_inputs : pytest.fixture
        Fixture inherited from `astrodata.testing` with path to the input files.
    path_to_outputs : pytest.fixture
        Fixture inherited from `astrodata.testing` with path to the output files.

    Returns
    -------
    namespace
        An object that contains `.input_dir` and `.output_dir`
    """
    oldmask = os.umask(000)  # Allows manipulating permissions

    # Define the ConfigTest class ---
    class ConfigTest:
        """
        Config class created for each dataset file. It is created from within
        this a fixture so it can inherit the `path_to_*` fixtures as well.
        """
        def __init__(self, path):

            log_dir = "./logs"

            dataset = sorted(
                glob.glob(os.path.join(path_to_inputs, path, '*.fits')))

            list_of_bias = dataselect.select_data(dataset, ['BIAS'], [])
            list_of_flats = dataselect.select_data(dataset, ['FLAT'], [])
            list_of_arcs = dataselect.select_data(dataset, ['ARC'], [])
            list_of_science = dataselect.select_data(dataset, [], ['CAL'])

            full_path = os.path.join(path_to_outputs, path)

            os.makedirs(log_dir, mode=0o775, exist_ok=True)
            os.makedirs(full_path, mode=0o775, exist_ok=True)

            config_file_name = os.path.join(full_path, "calibration_manager.cfg")

            if os.path.exists(config_file_name):
                os.remove(config_file_name)

            config_file_content = (
                "[calibs]\n"
                "standalone = False\n"
                "database_dir = {:s}\n".format(full_path)
            )

            with open(config_file_name, mode='w') as config_file:
                config_file.write(config_file_content)
            os.chmod(config_file_name, mode=0o775)

            calibration_service = cal_service.CalibrationService()
            calibration_service.config(config_file=config_file_name)

            self.arcs = list_of_arcs
            self.biases = list_of_bias
            self.calibration_service = calibration_service
            self.flats = list_of_flats
            self.full_path = full_path
            self.log_dir = log_dir
            self.science = list_of_science

    # Create ConfigTest object ---
    c = ConfigTest(request.param)
    yield c

    # Tear Down ---
    for root, dirs, files in os.walk('calibrations/'):
        for f in files:
            os.chmod(os.path.join(root, f), 0o775)
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o775)

    try:
        shutil.rmtree(os.path.join(c.full_path, 'calibrations/'))
    except FileNotFoundError:
        pass

    shutil.move('calibrations/', c.full_path)

    for f in glob.glob(os.path.join(os.getcwd(), '*.fits')):
        shutil.move(f, os.path.join(c.full_path, f))

    for root, dirs, files in os.walk(c.full_path):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o775)
        for f in files:
            os.chmod(os.path.join(root, f), 0o775)

    os.umask(oldmask)  # Restores default permission restrictions
    del c


@pytest.mark.gmosls
class TestGmosReduceLongslit:
    """
    Collection of tests that will run on every `dataset_folder`. Both
    `dataset_folder` and `calibrations` parameter should be present on every
    test. Even when the test does not use it.
    """
    @staticmethod
    def test_can_run_reduce_bias(config):
        """
        Make sure that the reduce_BIAS works for spectroscopic data.
        """
        logutils.config(
            mode='quiet', file_name=os.path.join(
                config.log_dir, 'reduce_GMOS_LS_bias.log'))

        reduce = Reduce()
        reduce.files.extend(config.biases)
        reduce.upload = 'calibs'
        reduce.runr()

    @staticmethod
    def test_can_run_reduce_flat(config):
        """
        Make sure that the reduce_FLAT_LS_SPECT works for spectroscopic data.
        """
        logutils.config(
            mode='quiet', file_name=os.path.join(
                config.log_dir, 'reduce_GMOS_LS_flat.log'))

        reduce = Reduce()
        reduce.files.extend(config.flats)
        reduce.upload = 'calibs'
        reduce.runr()

    @staticmethod
    @pytest.mark.skip(reason="Waiting dev fitsserver to be deployed")
    def test_can_run_reduce_arc(config):
        """
        Make sure that the recipes_ARC_LS_SPECT can run for spectroscopic
        data.
        """
        logutils.config(
            mode='quiet', file_name=os.path.join(
                config.log_dir, 'reduce_GMOS_LS_arc.log'))

        reduce = Reduce()
        reduce.files.extend(config.arcs)
        reduce.upload = 'calibs'
        reduce.runr()

    @staticmethod
    @pytest.mark.skip(reason="Define first how flats are processed")
    def test_can_run_reduce_science(config):
        """
        Make sure that the recipes_ARC_LS_SPECT works for spectroscopic data.
        """
        logutils.config(
            mode='quiet', file_name=os.path.join(
                config.log_dir, 'reduce_GMOS_LS_science.log'))

        reduce = Reduce()
        reduce.files.extend(config.science)
        reduce.upload = 'calibs'
        reduce.runr()


if __name__ == '__main__':
    pytest.main()
