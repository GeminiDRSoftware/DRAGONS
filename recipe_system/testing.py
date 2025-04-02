import os
import pytest

import astrodata
import gemini_instruments

from astrodata import testing
from gempy.utils import logutils
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals


@pytest.fixture(scope='module')
def get_master_arc(path_to_inputs, change_working_dir):
    """
    Factory that creates a function that reads the master arc file from the
    permanent input folder or from the temporarily local cache, depending on
    command line options.

    Parameters
    ----------
    path_to_inputs : pytest.fixture
        Path to the permanent local input files.
    change_working_dir : contextmanager
        Enable easy change to temporary folder when reducing data.

    Returns
    -------
    AstroData
        The master arc.
    """

    def _get_master_arc(ad, pre_process):

        cals = testing.get_associated_calibrations(
            ad.filename.split('_')[0] + '.fits')

        arc_filename = cals[cals['caltype'] == 'arc']['filename'][0]
        arc_filename = arc_filename.split('.fits')[0] + '_arc.fits'

        if pre_process:
            with change_working_dir():
                master_arc = astrodata.from_file(arc_filename)
        else:
            master_arc = astrodata.from_file(
                os.path.join(path_to_inputs, arc_filename))

        return master_arc

    return _get_master_arc


@pytest.fixture(scope='module')
def reduce_arc(change_working_dir):
    """
    Factory for function for ARCS data reduction.

    Parameters
    ----------
    change_working_dir : pytest.fixture
        Context manager used to write reduced data to a temporary folder.

    Returns
    -------
    function : A function that will read the arcs files, process them and
    return the name of the master arc.
    """

    def _reduce_arc(dlabel, arc_fnames):
        with change_working_dir():
            print("Reducing ARCs in folder:\n  {}".format(os.getcwd()))
            # Use config to prevent duplicated outputs when running Reduce via API
            logutils.config(file_name='log_arc_{}.txt'.format(dlabel))

            reduce = Reduce()
            reduce.files.extend(arc_fnames)
            reduce.runr()

            master_arc = reduce.output_filenames.pop()
        return master_arc

    return _reduce_arc


@pytest.fixture(scope='module')
def reduce_bias(change_working_dir):
    """
    Factory for function for BIAS data reduction.

    Parameters
    ----------
    change_working_dir : pytest.fixture
        Context manager used to write reduced data to a temporary folder.

    Returns
    -------
    function : A function that will read the bias files, process them and
    return the name of the master bias.
    """

    def _reduce_bias(datalabel, bias_fnames):
        with change_working_dir():
            print("Reducing BIAS in folder:\n  {}".format(os.getcwd()))
            logutils.config(file_name='log_bias_{}.txt'.format(datalabel))

            reduce = Reduce()
            reduce.files.extend(bias_fnames)
            reduce.runr()

            master_bias = reduce.output_filenames.pop()

        return master_bias

    return _reduce_bias


@pytest.fixture(scope='module')
def reduce_flat(change_working_dir):
    """
    Factory for function for FLAT data reduction.

    Parameters
    ----------
    change_working_dir : pytest.fixture
        Context manager used to write reduced data to a temporary folder.

    Returns
    -------
    function : A function that will read the flat files, process them and
    return the name of the master flat.
    """

    def _reduce_flat(data_label, flat_fnames, master_bias):
        with change_working_dir():
            print("Reducing FLATs in folder:\n  {}".format(os.getcwd()))
            logutils.config(file_name='log_flat_{}.txt'.format(data_label))

            calibration_files = ['processed_bias:{}'.format(master_bias)]

            reduce = Reduce()
            reduce.files.extend(flat_fnames)
            reduce.mode = 'ql'
            reduce.ucals = normalize_ucals(calibration_files)
            reduce.runr()

            master_flat = reduce.output_filenames.pop()
            master_flat_ad = astrodata.from_file(master_flat)

        return master_flat_ad

    return _reduce_flat


@pytest.fixture(scope="module")
def ref_ad_factory(path_to_refs):
    """
    Read the reference file.

    Parameters
    ----------
    path_to_refs : pytest.fixture
        Fixture containing the root path to the reference files.

    Returns
    -------
    function : function that loads the reference file.
    """

    def _reference_ad(filename):
        print(f"Loading reference file: {filename}")
        path = os.path.join(path_to_refs, filename)
        return astrodata.from_file(path)

    return _reference_ad

