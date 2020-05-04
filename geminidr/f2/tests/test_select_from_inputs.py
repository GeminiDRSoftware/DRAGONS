#!/usr/bin/env python

import glob
import os
import pytest

import astrodata, gemini_instruments

from geminidr.f2.primitives_f2_image import F2Image


@pytest.mark.skip(reason="Test is hanging out forever. Needs to be reviewed.")
def test_select_from_inputs_primitive(path_to_inputs):

    with open("recursion.log", 'w') as _log:

        for f in glob.glob(os.path.join(path_to_inputs, 'F2', '*.fits')):

            ad = astrodata.open(f)

            p = F2Image([ad])

            p.prepare()
            p.addVAR(read_noise=True)

            try:

                print('Running tests on file: {:s}'.format(f))
                p.selectFromInputs(tags="DARK", outstream="darks")
                p.showInputs(stream="darks")
                p.showInputs()

                print("{:15s} OK".format(f))
                _log.write("\n{:15s} OK".format(f))

            except RecursionError as re:

                print("{:15s} FAIL".format(f))
                _log.write("\n{:15s} FAIL".format(f))

            del ad
            del p


if __name__ == '__main__':

    pytest.main()
