#!/usr/bin/env python

import glob
import pytest
import os

import astrodata
import gemini_instruments
import geminidr

from gempy.adlibrary import dataselect
from geminidr.f2.recipes.sq.recipes_DARK import makeProcessedDark


@pytest.fixture(scope='module')
def caldb():

    from recipe_system.cal_service import set_calservice, CalibrationService

    calibration_service = CalibrationService()
    calibration_service.config()
    calibration_service.init()

    set_calservice()

    return calibration_service


def test_reduce_darks(test_path, caldb):

    all_files = glob.glob(os.path.join(test_path, 'F2', '*.fits'))

    expression = 'exposure_time==3'
    parsed_expr = dataselect.expr_parser(expression)
    selected_files = dataselect.select_data(
        all_files, ['F2', 'DARK'], [], parsed_expr)

    dataset = [astrodata.open(f) for f in selected_files]

    p = geminidr.f2.primitives_f2.F2(dataset)
    makeProcessedDark(p)


if __name__ == '__main__':
    pytest.main()

