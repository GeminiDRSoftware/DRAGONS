import os

import numpy as np
import psutil
import pytest

import astrodata

from gempy.utils import logutils
from geminidr.core import primitives_visualize


@pytest.fixture
def astrofaker():
    try:
        import astrofaker
    except ImportError:
        pytest.skip("astrofaker not installed")

    return astrofaker


def test_mosaic_detectors_gmos_binning(astrofaker):
    """
    Tests that the spacing between amplifier centres for NxN binned data
    is precisely N times smaller than for unbinned data when run through
    mosaicDetectors()
    """
    from geminidr.gmos.primitives_gmos_image import GMOSImage

    for hemi in 'NS':
        for ccd in ('EEV', 'e2v', 'Ham'):
            for binning in (1, 2, 4):
                try:
                    ad = astrofaker.create('GMOS-{}'.format(hemi), ['IMAGE', ccd])
                except ValueError:  # No e2v for GMOS-S
                    continue
                ad.init_default_extensions(binning=binning, overscan=False)
                for ext in ad:
                    shape = ext.data.shape
                    ext.add_star(amplitude=10000, x=0.5 * (shape[1] - 1),
                                 y=0.5 * (shape[0] - 1), fwhm=0.5 * binning)
                p = GMOSImage([ad])
                ad = p.mosaicDetectors([ad])[0]
                ad = p.detectSources([ad])[0]
                x = np.array(sorted(ad[0].OBJCAT['X_IMAGE']))
                if binning == 1:
                    unbinned_positions = x
                else:
                    diffs = np.diff(unbinned_positions) - binning * np.diff(x)
                    assert np.max(abs(diffs)) < 0.01


@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("fname, arc_fname", zip(["N20180112S0209.fits"], ["N20180112S0353.fits"]))
def test_plot_spectra_for_qa_single_frame(fname, arc_fname, path_to_inputs):

    def process_arc(filename, suffix="distortionDetermined"):
        """
        Helper recipe to reduce the arc file.

        Returns
        -------
        AstroData
            Processed arc.
        """
        from astrodata.testing import download_from_archive
        from geminidr.gmos.primitives_gmos_longslit import GMOSLongslit

        processed_filename, ext = os.path.splitext(filename)
        processed_filename += "_{:s}{:s}".format(suffix, ext)

        if os.path.exists(processed_filename):
            ad = astrodata.open(processed_filename)
        else:

            if os.path.exists(filename):
                ad = astrodata.open(filename)
            else:
                ad = astrodata.open(download_from_archive(
                    filename, path='', env_var='DRAGONS_TEST'))

            p = GMOSLongslit([ad])

            p.prepare()
            p.addDQ(static_bpm=None)
            p.addVAR(read_noise=True)
            p.overscanCorrect()
            # p.biasCorrect()
            p.ADUToElectrons()
            p.addVAR(poisson_noise=True)
            p.mosaicDetectors()
            p.makeIRAFCompatible()
            p.determineWavelengthSolution()
            p.determineDistortion()

            ad = p.streams['main'][0]
            ad.write(overwrite=True)

        return ad

    def process_object(filename, arc, suffix="linearized"):
        """
        Helper recipe to reduce the object file.

        Returns
        -------
        AstroData
            Processed arc.
        """
        from astrodata.testing import download_from_archive
        from geminidr.gmos.primitives_gmos_longslit import GMOSLongslit

        processed_filename, ext = os.path.splitext(filename)
        processed_filename += "_{:s}{:s}".format(suffix, ext)

        if os.path.exists(processed_filename):
            ad = astrodata.open(processed_filename)
        else:

            if os.path.exists(filename):
                ad = astrodata.open(filename)
            else:
                ad = astrodata.open(download_from_archive(
                    filename, path='', env_var='DRAGONS_TEST'))

            p = GMOSLongslit([ad])

            p.prepare()
            p.addDQ(static_bpm=None)
            p.addVAR(read_noise=True)
            p.overscanCorrect()
            # p.biasCorrect()
            p.ADUToElectrons()
            p.addVAR(poisson_noise=True)
            # p.flatCorrect()
            # p.applyQECorrection()
            p.distortionCorrect(arc=arc)
            p.findSourceApertures(max_apertures=1)
            p.skyCorrectFromSlit()
            p.traceApertures()
            p.extract1DSpectra()
            p.linearizeSpectra()  # TODO: needed?
            p.calculateSensitivity()

            ad = p.streams['main'][0]
            ad.write(overwrite=True)

        return ad

    def check_if_process_running(process_name):
        """
        Check if there is any running process that contains the given name processName.
        """
        for proc in psutil.process_iter():
            try:
                if process_name.lower() in proc.name().lower():
                    return True

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        return False


    if not check_if_process_running("adcc"):
        pytest.skip("ADCC is not running.")
    else:
        print("ADCC is up and running!")

    os.chdir(path_to_inputs)
    print(' Current working directory:\n   {}'.format(path_to_inputs))

    logutils.config("standard",
                    file_name=os.path.basename(__file__.replace('.py', '.log')))

    arc_ad = process_arc(arc_fname)
    ad = process_object(fname, arc=arc_ad)

    p = primitives_visualize.Visualize([])
    p.plotSpectraForQA(adinputs=[ad])
