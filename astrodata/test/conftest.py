
import pytest
import os


def pytest_addoption(parser):

    parser.addoption(
        "--ad_test_data_path",
        type=str,
        default=None,
        help="Local path to where the test data should live."
    )


def pytest_collection_modifyitems(config, items):

    local_path = config.getoption("--ad_test_data_path")

    if os.path.exists(local_path):
        return

        skip_ad_local_tests = pytest.mark.skip(
            reason="need --ad_test_data_path option to run")
    else:
        skip_ad_local_tests = pytest.mark.skip(
            reason="input --ad_test_data_path does not exists")

    for item in items:
        if "ad_local_data" in item.keywords:
            item.add_marker(skip_ad_local_tests)


@pytest.fixture
def test_path(request):
    """
    Returns the path that contains any FITS image that would be used for
    testing.
    """
    return request.config.getoption("--ad_test_data_path")

