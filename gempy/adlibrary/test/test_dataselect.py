#!/usr/bin/env python

import glob
import pytest
import os

from gempy.adlibrary import dataselect
from recipe_system.reduction.coreReduce import Reduce

from gempy.utils import logutils

logutils.config(file_name='dummy.log')


@pytest.fixture
def test_path():

    try:
        path = os.environ['TEST_PATH']
    except KeyError:
        pytest.skip("Could not find environment variable: $TEST_PATH")

    if not os.path.exists(path):
        pytest.skip("Could not find path stored in $TEST_PATH: {}".format(path))

    return path


def test_expr_parser_can_parse_for_exposure_time():
    """
    Does exactly what expr_parser does when the requested descriptor is
    'exposure_time', but a more pythonic way. Should always return the same
    value as expr_parser when working with exposure_times, but
    will not work for operations not defined (Ones not '!=<>')
    """
    expression = ('filter_name=="Some_Filter"', 'filter_name=="Other_Filter',
                  'filter_name>"small_fliter"', 'filter_name<"Big_Filter',
                  'filter_name!"Bad_Filter')

    for value in expression:
        # makes sure the expression starts with a 'filter_name' as this
        # test only focuses on these cases.
        assert "filter_name" in value

        # resets following strings to 0 so their value from the last iter
        descriptor_answer = descriptor = used_operator = None

        for operator in "!=<>":
            if operator in value:

                descriptor = value.split(operator)[0]
                descriptor_answer = value.split(operator)[-1]
                used_operator = operator

                if operator is "=":
                    # = -> == from testing to asserting in string
                    used_operator = "=="

        assert used_operator is not None

        expected = 'ad.' + descriptor + '(pretty=True)' \
                    + used_operator + descriptor_answer
        answer = dataselect.expr_parser(value)

        assert answer == expected

    #   expr_parser('filter_name=="Kshort"')