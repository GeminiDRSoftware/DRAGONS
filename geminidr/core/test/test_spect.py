#!/usr/bin/env python

import glob
import pytest
import os
import warnings

import astrodata
import astrodata.testing as ad_test
import gemini_instruments
import geminidr

from gempy.utils import logutils


logutils.config(file_name='foo.log')


def prepare_for_wavelength_calibration(p):
    """
    Runs the primitives that come before running the actual wavelength calibration
    one.

    Parameters
    ----------
        p : :class:`~geminidr.core.primitives_spect.Spect` or subclass

    Returns
    -------
        :class:`~geminidr.core.primitives_spect.Spect` or subclass : returns
        the same object that entered this function after processing the data.
    """
    p.viewer = geminidr.dormantViewer(p, None)
    p.prepare()
    p.addDQ(static_bpm=None)
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.mosaicDetectors()
    p.makeIRAFCompatible()

    return p


@pytest.mark.parametrize("input_file", ["GMOS/GN-2017A-FT-19/N20170530S0006.fits"])
def test_determine_wavelength_solution(input_file, path_to_inputs, path_to_outputs, path_to_refs):
    """
    Regression test for determine wavelength solution.
    """
    _input = os.path.join(path_to_inputs, input_file)
    _output = os.path.join(path_to_outputs, input_file)
    _reference = os.path.join(path_to_refs, input_file)

    if not os.path.exists(_input):
        pytest.mark.xfail(reason="Could not access input file:\n"
                                 "     {}".format(_input))

    if not os.path.exists(os.path.dirname(_output)):
        os.makedirs(os.path.dirname(_output), exist_ok=True)

    if not os.path.exists(os.path.dirname(_reference)):
        os.makedirs(os.path.dirname(_reference), exist_ok=True)

    if os.path.exists(_reference):
        reference_ad = astrodata.open(_reference)
    else:
        warnings.warn("Could not access reference file:\n    {}".format(
            _reference))
        reference_ad = astrodata.open(_output)

    ad = astrodata.open(_input)

    if 'GMOS' in ad.tags:
        from geminidr.gmos import primitives_gmos_spect
        p = primitives_gmos_spect.GMOSSpect([ad])
    else:
        pytest.fail("Input data tags do not match expected tags for tests.\n"
                    "    {}".format(ad.tags))

    p = prepare_for_wavelength_calibration(p)
    p.determineWavelengthSolution()
    p.writeOutputs(outfilename=_output)

    output_ad = astrodata.open(_output)

    ad_test.assert_same_class(output_ad, reference_ad)
    ad_test.assert_wavelength_solutions_are_close(output_ad, reference_ad)

    output_ad.write(filename=_reference, overwrite=True)


if __name__ == '__main__':
    pytest.main()
