import pytest
import os.path

from astrodata.test import conftest

input_test_path = conftest.input_test_path
output_test_path = conftest.output_test_path


@pytest.fixture
def path_to_raw_files(input_test_path):

    return os.path.join(input_test_path, 'raw')

@pytest.fixture
def path_to_ref_files(input_test_path):

    return os.path.join(input_test_path, 'ref')

@pytest.fixture
def proc_dir(output_test_path):

    # Append proc/ to environment-defined path unless we're just using the CWD:
    if output_test_path:
        path = os.path.join(output_test_path, 'proc')

    return path

