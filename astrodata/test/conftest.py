
import pytest
import os
import glob

#@pytest.fixture
def test_path():

    try:
        path = os.environ['TEST_PATH']
    except KeyError:
        pytest.skip("Could not find environment variable: $TEST_PATH")

    if not os.path.exists(path):
        pytest.skip("Could not find path stored in $TEST_PATH: {}".format(path))

    return path

#
# def pytest_generate_tests(metafunc):
#     if 'test_file' in metafunc.fixturenames:
#         print(dir(metafunc))
#         try:
#             path = os.environ['TEST_PATH']
#         except KeyError:
#
#             pytest.mark.skip(metafunc, reason="Could not find environment variable: $TEST_PATH")
#
#         #
#         # if not os.path.exists(path):
#         #     pytest.skip("Could not find path stored in $TEST_PATH: {}".format(path))
#         #testfiles = glob.glob(os.path.join(test_path(), "*.fits"))
#
#         testfiles = ['a', 'b', 'c']
#         metafunc.parametrize('test_file', testfiles)
#
