#!/usr/bin/env python
"""
Tests applied to primitives_spect.py

"""
import pytest
import os
import numpy as np

import astrodata
import astrodata.testing as ad_test
import gemini_instruments
import geminidr

from gempy.utils import logutils

logutils.config(file_name='foo.log')

file_list = [
    "process_arcs/N20170530S0006.fits",
    "process_arcs/N20180119S0232.fits",
    "process_arcs/N20181114S0512.fits",
    "process_arcs/N20180120S0417.fits",
    # "process_arcs/N20180911S0156.fits", # File failed loading
    # "process_arcs/S20181022S0349.fits", # File failed loading
    # "process_arcs/N20181011S0220.fits", # File failed loading
    # "process_arcs/N20181115S0175.fits", # File failed loading
    # "process_arcs/S20181128S0099.fits", # File failed loading
]


@pytest.fixture(params=file_list)
def test_files(request, path_to_inputs, path_to_outputs, path_to_refs):
    """
    Fixture that parses the `file_list` parameter list into a Namespace in a
    cleaner fashion. A single fixture is created in order to avoid a parameter
    matrix that would be created if one fixture was done for each variable.

    Parameters
    ----------
    request :
        This is an internal variable from PyTest and it is used to access each
        parameter in the parameter list.

    path_to_inputs : str
        This is the path to where the input files are stored. It is accessed
        via `path_to_inputs` fixture, defined in the conftest.py file

    path_to_outputs : str
        This is the path to where the outputs files are stored. It is accessed
        via `path_to_outputs` fixture, defined in the conftest.py file

    path_to_refs : str
        This is the path to where the reference files are stored. It is accessed
        via `path_to_refs` fixture, defined in the conftest.py file

    Returns
    -------
    Namespace :
        An object that holds the full path for the input, output and reference
        files.
    """
    class TestFiles:
        pass

    TestFiles.input = os.path.join(path_to_inputs, request.param)
    TestFiles.output = os.path.join(path_to_outputs, request.param)
    TestFiles.reference = os.path.join(path_to_refs, request.param)

    return TestFiles


def add_suffix(file_name, suffix):
    """
    Adds a suffix to the input file_name.

    Parameters
    ----------
    file_name : str
        Input file name

    suffix : str
        Suffix to be appended to the file name


    Returns
    -------
    str
        new file name with suffix properly appended
    """
    return "".join([os.path.splitext(file_name)[0], suffix, ".fits"])


def prepare_for_wavelength_calibration(ad):
    """
    Runs the primitives that come before running the actual wavelength calibration
    one.

    Parameters
    ----------
        ad : :class:`~astrodata.AstroData` or subclass

    Returns
    -------
        :class:`~geminidr.core.primitives_spect.Spect` or subclass : returns
        the same object that entered this function after processing the data.
    """

    if 'GMOS' in ad.tags:
        from geminidr.gmos import primitives_gmos_spect
        p = primitives_gmos_spect.GMOSSpect([ad])
    else:
        pytest.fail("Input data tags do not match expected tags for tests.\n"
                    "    {}".format(ad.tags))

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


def test_QESpline_optimization():
    """
    Test the optimization of the QESpline. This defines 3 regions, each of a
    different constant value, with gaps between them. The spline optimization
    should determine the relative offsets.
    """
    from geminidr.core.primitives_spect import QESpline
    GAP = 20
    DATA_LENGTH = 300
    real_coeffs = [0.5, 1.2]
    data = np.array([1] * DATA_LENGTH + [0] * GAP + [real_coeffs[0]] * DATA_LENGTH + [0] * GAP + [real_coeffs[1]] * DATA_LENGTH)
    xpix = np.arange(len(data))
    weights = np.where(data > 0, 1., 0.)
    boundaries = (DATA_LENGTH, 2*DATA_LENGTH+GAP)

    coeffs = np.ones((2,))
    order = 10
    result = QESpline(coeffs, xpix, data, weights, boundaries, order)
    np.testing.assert_allclose(real_coeffs, 1./result.x, atol=0.01)


class TestArcProcessing:
    """
    This class structure simply holds together tests inside a single context.
    There is no actual functionality.
    """

    def test_determine_wavelength_solution(self, test_files):
        """
        Regression test for determine wavelength solution. It checks if the
        :class:`~astrodata.AstroData` object contains extension(s) with the "WAVECAL"
        attribute and compare them.

        It processes an input raw file, writes the processed data to an output file
        and compare the file with a reference file.

        Parameters
        ----------
        test_files : Namespace
            An object that contains an `.input`, an `.output` and a `.reference`
            reference containing strings with the paths to the respective
            input, output and reference files.
        """
        _suffix = "_wavelength_solution_determined"

        _input = test_files.input
        _output = add_suffix(test_files.output, _suffix)
        _reference = add_suffix(test_files.reference, _suffix)

        if not os.path.exists(_input):
            pytest.mark.xfail(
                reason="Could not access input file:\n     {}".format(_input))

        if not os.path.exists(os.path.dirname(_output)):
            os.makedirs(os.path.dirname(_output), exist_ok=True)

        if not os.path.exists(os.path.dirname(_reference)):
            os.makedirs(os.path.dirname(_reference), exist_ok=True)

        ad = astrodata.open(_input)

        p = prepare_for_wavelength_calibration(ad)
        p.determineWavelengthSolution()
        p.writeOutputs(outfilename=_output, overwrite=True)

        output_ad = astrodata.open(_output)
        reference_ad = astrodata.open(_reference)

        ad_test.assert_same_class(output_ad, reference_ad)
        ad_test.assert_wavelength_solutions_are_close(output_ad, reference_ad)

    def test_determine_distortion(self, test_files):
        """
        Regression test for determine distortion.

        It processes an input raw file, writes the processed data to an output file
        and compare the file with a reference file.

        Parameters
        ----------
        test_files : Namespace
            An object that contains an `.input`, an `.output` and a `.reference`
            reference containing strings with the paths to the respective
            input, output and reference files.
        """
        _suffix = "_distortionDetermined"

        _input = test_files.input
        _output = add_suffix(test_files.output, _suffix)
        _reference = add_suffix(test_files.reference, _suffix)

        if not os.path.exists(_input):
            pytest.mark.xfail(
                reason="Could not access input file:\n     {}".format(_input))

        if not os.path.exists(os.path.dirname(_output)):
            os.makedirs(os.path.dirname(_output), exist_ok=True)

        if not os.path.exists(os.path.dirname(_reference)):
            os.makedirs(os.path.dirname(_reference), exist_ok=True)

        ad = astrodata.open(_input)

        p = prepare_for_wavelength_calibration(ad)
        p.determineWavelengthSolution()
        p.determineDistortion()
        p.writeOutputs(outfilename=_output, overwrite=True)

        output_ad = astrodata.open(_output)
        reference_ad = astrodata.open(_reference)

        ad_test.assert_same_class(output_ad, reference_ad)
        ad_test.assert_have_same_distortion(output_ad, reference_ad)


if __name__ == '__main__':
    pytest.main()
