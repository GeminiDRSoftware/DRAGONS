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


def test_expr_parser_can_parse_for_filter_name():
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


def test_expr_parser_can_parse_for_exposure_time():

    expression = ('exposure_time==60.', 'exposure_time==-45',
                  'exposure_time > 30.', 'exposure_time < 40.3',
                  'exposure_time! 60')

    for value in expression:
        # makes sure the expression starts with a 'filter_name' as this
        # test only focuses on these cases.
        assert "exposure_time" in value

        # resets following strings to 0 so their value from the last iter
        descriptor_answer = descriptor = used_operator = None

        for operator in "!=<>":
            if operator in value:
                descriptor = value.split(operator)[0]
                descriptor_answer = value.split(operator)[-1]
                used_operator = operator

        assert used_operator is not None

        # If operator is not '==', then original codified expression is returned
        if used_operator is not "=":
            expected = descriptor + used_operator + descriptor_answer

        else:
            expected = 'isclose(ad.' + descriptor + '(),' \
                       + descriptor_answer + ")"

        # Actual answer we're comparing to
        answer = dataselect.expr_parser(value)

        assert answer == expected


def test_expr_parser_can_parse_for_ut_datetime():
    """
    Does exactly what expr_parser does when the requested descriptor is
    'ut_datetime', but a more pythonic way. Should always return the same
    value as expr_parser when working with ut_datetime, but
    will not work for operations not defined (Ones not '!=<>')
    """
    expression = ('ut_datetime=="Some_time"', 'ut_datetime==1998-02-23-12-43-43-52',
                  'ut_datetime>2017-04-07-06-04-03', 'ut_datetime<2007-05-23-16-54-01',
                  'ut_datetime!2015-02-20-17-14-03')

    for value in expression:
        # makes sure the expression starts with a 'ut_datetime' as this
        # test only focuses on these cases.
        assert "ut_datetime" in value

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

        expected = 'ad.' + descriptor + '()' + used_operator \
                   + 'datetime.strptime(' + descriptor_answer \
                   + ', "%Y-%m-%d %H:%M:%S")'
        answer = dataselect.expr_parser(value)

        assert answer == expected


def test_expr_parser_can_parse_for_ut_date():
    """
    Does exactly what expr_parser does when the requested descriptor is
    'ut_date', but a more pythonic way. Should always return the same
    value as expr_parser when working with ut_date, but
    will not work for operations not defined (Ones not '!=<>')
    """
    expression = ('ut_date==1999-12-13"', 'ut_date==2011-03-14',
                  'ut_date>2018-12-31', 'ut_date<2001-01-01',
                  'ut_date!2018-08-31')

    for value in expression:
        # makes sure the expression starts with a 'ut_date' as this
        # test only focuses on these cases.
        assert "ut_date" in value

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

        expected = 'ad.' + descriptor + '()' + used_operator \
                   + 'datetime.strptime(' + descriptor_answer \
                   + ', "%Y-%m-%d").date()'
        answer = dataselect.expr_parser(value)

        assert answer == expected


def test_expr_parser_can_parse_for_ut_time_or_local_time():
    """
    Does exactly what expr_parser does when the requested descriptor is
    'ut_time' or 'local_time', but a more pythonic way. Should always return the same
    value as expr_parser when working with ut_time_or_local_time, but
    will not work for operations not defined (Ones not '!=<>')
    """
    expression = ('ut_time==22-4-14', 'ut_time==31-14-54',
                  'ut_time>12-15-43', 'ut_time<07-41-41',
                  'ut_time!12-0-0',
                  'local_time=="Some_Filter"', 'local_time=="Other_Filter',
                  'local_time>"small_fliter"', 'local_time<"Big_Filter',
                  'local_time!"Bad_Filter'
                  )

    for value in expression:
        # makes sure the expression starts with a 'ut_time_or_local_time' as this
        # test only focuses on these cases.
        assert ("ut_time" in value) or ("local_time" in value)

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

        expected = 'ad.' + descriptor + '()' + used_operator \
                   + 'datetime.strptime(' + descriptor_answer \
                   + ', "%H:%M:%S").time()'
        answer = dataselect.expr_parser(value)

        assert answer == expected
