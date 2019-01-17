#!/usr/bin/env python

import astrodata
import gemini_instruments
import os
import pytest

from copy import deepcopy
from glob import glob

from geminidr.f2.primitives_f2_image import F2Image


TEST_PATH = os.environ['TEST_PATH']


def test_select_from_inputs_primitive():

    with open("recursion.log", 'w') as _log:

        for f in glob(os.path.join(TEST_PATH, 'F2', '*.fits')):

            ad = astrodata.open(f)

            p = F2Image([ad])

            p.prepare()
            p.addVAR(read_noise=True)

            try:

                p.selectFromInputs(tags="DARK", outstream="darks")
                p.showInputs(stream="darks")
                p.showInputs()

                print("{:15s} OK".format(f))
                _log.write("{:15s} OK".format(f))

            except RecursionError as re:

                print("{:15s} FAIL".format(f))
                _log.write("{:15s} FAIL".format(f))

            del ad
            del p

if __name__ == '__main__':

    pytest.main()
