#!/usr/bin/env python

import pytest

# -- Tests --------------------------------------------------------------------
@pytest.mark.gmosls
@pytest.mark.dragons_remote_data
def test_applied_qe_is_locally_continuous():
    pass


@pytest.mark.gmosls
@pytest.mark.dragons_remote_data
def test_applied_qe_is_globally_continuous():
    pass


@pytest.mark.gmosls
@pytest.mark.dragons_remote_data
def test_applied_qe_is_stable():
    pass


if __name__ == '__main__':
    pytest.main()
