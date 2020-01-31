#!/usr/bin/python
"""
Mock plotSpectraForQA primitive.

This module emulates a pipeline that reproduces how the plotSpectraForQA would
behave in real time.

In practice, it creates JSON data and send it to the ADCC Server in a timed loop.
"""
import json
import numpy as np
import time
import urllib

from scipy import ndimage

URL = "http://localhost:8777/spec_report"


def main():
    np.random.seed(0)

    n_frames = 3
    n_apertures = 3
    sleep_between_frames = 10.

    data_size = 4000
    snr = 10.
    wavelength_min = 300.
    wavelength_max = 800.
    wavelength_units = "nm"

    file_index = 1
    program_index = 1

    while True:

        obj_max_weight = 1000.
        obj_continnum = 300. + 0.01 * np.arange(data_size)

        noise_level = obj_continnum / snr

        wavelength = np.linspace(wavelength_min, wavelength_max, data_size)
        dispersion = np.mean(np.diff(wavelength))

        center = np.random.randint(100, 900, size=n_apertures)
        lower = np.random.randint(-15, -1, size=n_apertures)
        upper = np.random.randint(1, 15, size=n_apertures)

        data = [create_1d_spectrum(data_size, 20, obj_max_weight)
                for i in range(n_apertures)]

        year = 2020
        today = 20200131

        program_id = "GX-{}C-Q-{:03d}".format(year, program_index)

        group_index = 1
        group_id = "{:s}-{:02d}".format(program_id, group_index)

        for frame_index in range(n_frames):

            data_label = "{:s}-{:03d}".format(group_id, frame_index+1)
            filename = "X{}S{:03d}_frame.fits".format(today, file_index)

            is_stack = False

            def aperture_generator(i):
                _data = data[i]
                _error = np.random.poisson(_data)
                _aperture = ApertureModel(
                    center[i], lower[i], upper[i], wavelength_units, dispersion,
                    wavelength, _data, _error).__dict__
                yield _aperture

            # apertures = [aperture_generator(i) for i in range(n_apertures)]
            apertures = []

            frame = SpecPackModel(
                data_label, group_id, filename, is_stack, program_id, apertures)

            json_data = json.dumps([frame.__dict__]).encode("utf-8")

            print("\n Created JSON for single frame with: ")
            print("  Program ID: {}".format(program_id))
            print("  Group-id: {}".format(group_id))
            print("  Data-label: {}".format(data_label))
            print("  Filename: {}".format(filename))
            print("  Performing request...")

            post_request = urllib.request.Request(URL)
            postr = urllib.request.urlopen(post_request, json_data)
            postr.read()
            postr.close()

            print("  Done.")
            print("  Sleeping...")

            time.sleep(sleep_between_frames)
            file_index += 1

        program_index += 1
        group_index += 1


def create_1d_spectrum(width, n_lines, max_weight):
    """
    Generates a 1D NDArray that simulates a random spectrum.

    Parameters
    ----------
    width : int
        Number of array elements.
    n_lines : int
        Number of artificial lines.
    max_weight : float
        Maximum weight (or flux, or intensity) of the lines.

    Returns
    -------
    sky_1d_spectrum : numpy.ndarray

    """
    lines = np.random.randint(low=0, high=width, size=n_lines)
    weights = max_weight * np.random.random(size=n_lines)

    spectrum = np.zeros(width)
    spectrum[lines] = weights
    spectrum = ndimage.gaussian_filter1d(spectrum, 5)

    return spectrum


class ApertureModel:

    def __init__(self, center, lower, upper, wavelength_units, dispersion, wavelength, intensity, error):
        wavelength = np.round(wavelength, 3)
        intensity = np.round(intensity)
        error = np.round(error)

        self.center = center
        self.lower = lower
        self.upper = upper
        self.dispersion = dispersion
        self.wavelength_units = wavelength_units
        self.intensity = [[w, int(d)] for w, d in zip(wavelength, intensity)]
        self.stddev = [[w, int(d)] for w, d in zip(wavelength, error)]


class SpecPackModel:

    def __init__(self, data_label, group_id, filename, is_stack, program_id, apertures):
        self.data_label = data_label
        self.group_id = group_id
        self.filename = filename
        self.is_stack = is_stack
        self.program_id = program_id
        self.apertures = apertures


if __name__ == '__main__':
    main()
