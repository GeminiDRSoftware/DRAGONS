#!/usr/bin/env python
import pytest


datasets = [
    "GS-2018A-Q-117-69",  # B600
    "GS-2019A-Q-111-23",  # B600
    "GS-2019B-Q-222-305",  # B600
]


@pytest.mark.gmosls
def test_processed_flat_has_median_around_one():
    pass


@pytest.mark.gmosls
def test_processed_flat_is_stable():
    pass


if __name__ == '__main__':
    pytest.main()
