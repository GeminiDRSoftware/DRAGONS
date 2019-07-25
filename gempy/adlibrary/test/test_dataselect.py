#!/usr/bin/env python

import glob
import pytest
import os

from gempy.adlibrary import dataselect
from recipe_system.reduction.coreReduce import Reduce

from gempy.utils import logutils

logutils.config(file_name='dummy.log')


try:
    path = os.environ['TEST_PATH']
except KeyError:
    pytest.skip("Could not find environment variable: $TEST_PATH")

if not os.path.exists(path):
    pytest.skip("Could not find path stored in $TEST_PATH: {}".format(path))


# Returns list of all files in the TEST_PATH directory
F2_reduce = glob.glob(os.path.join(path, "F2/test_reduce/", "*fits"))

# Separates the directory from the list, helps cleanup code
fits_files = [_file.split('/')[-1] for _file in F2_reduce]
assert len(fits_files) > 1


def test_isclose_returns_proper_values_with_edgecases():
    """
    Two lists, one list that calls 'isclose()', other has list of bools
    which correspond to the expected answer. This just means we can shorten
    the amount of lines in this test and use one assert instead of multiple
    """
    answer_values = [dataselect.isclose(1.0, 2.0, 2.0), dataselect.isclose(1.0, 2.1),
                     dataselect.isclose(0.1, 0.1, 0.0), dataselect.isclose(0.1, 0.11),
                     dataselect.isclose(1.5, 3.0, 1.4), dataselect.isclose(4, 8, 0.1, 9)]
    excpected_values = [True, False, True, False, True, True]
    for i in range(len(answer_values)):
        assert answer_values[i] == excpected_values[i]


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


def test_evalexpression():
    """
    Tests  bunch of examples that could be used with evalexpression
    making sure they all assert to the proper bool
    """

    answer0 = dataselect.evalexpression("ad", "True")
    answer1 = dataselect.evalexpression("ad", "1 > 0")
    answer2 = dataselect.evalexpression("ad", "0 == 0")
    answer3 = dataselect.evalexpression("ad", "not(0 != 0)")
    assert answer0 is answer1 is answer2 is answer3 is True

    answer0 = dataselect.evalexpression("ad", "1 < 0")
    answer1 = dataselect.evalexpression("ad", "False")
    answer2 = dataselect.evalexpression("ad", "0 != 0")
    answer3 = dataselect.evalexpression("ad", "not(0 == 0)")
    assert answer0 is answer1 is answer2 is answer3 is False


def test_select_data():
    answer = dataselect.select_data(F2_reduce, ["F2", "FLAT"], [],
                                    dataselect.expr_parser('filter_name=="Y"'))

    # For legibility, the list of answers just has the files, we append this to
    # the full path in the for loop
    correct_files = ['S20131126S1113.fits', 'S20131129S0322.fits',
                     'S20131126S1111.fits', 'S20131129S0321.fits',
                     'S20131126S1115.fits', 'S20131126S1116.fits',
                     'S20131129S0323.fits', 'S20131126S1112.fits',
                     'S20131129S0320.fits', 'S20131126S1114.fits']

    for i in range(len(correct_files)):
        # adds the full path to the files above and makes sure each value
        # in the select_data instance is the correct string.
        expected = os.path.join(path, "F2/test_reduce/", correct_files[i])
        assert answer[i] == expected
