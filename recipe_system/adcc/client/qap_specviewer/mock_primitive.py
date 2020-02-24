#!/usr/bin/env python
"""
Mock plotSpectraForQA primitive.

This module emulates a pipeline that reproduces how the plotSpectraForQA would
behave in real time.

In practice, it creates JSON data and send it to the ADCC Server in a timed loop.
"""
import json
import time
import urllib.error
import urllib.request

import numpy as np
from scipy import ndimage

URL = "http://localhost:8777/spec_report"


def main():
    """
    Main function.
    """
    np.random.seed(0)
    args = _parse_arguments()

    url = args.url
    n_frames = args.n_frames
    n_apertures = args.n_apertures
    sleep_between_frames = args.sleep_time

    data_size = 4000
    snr = 10.
    pixel_scale = 0.1614
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

        data = [create_1d_spectrum(data_size, 20, obj_max_weight) + obj_continnum * i for i in range(n_apertures)]

        center = np.random.randint(100, 900, size=n_apertures+1)
        lower = np.random.randint(-15, -1, size=n_apertures+1)
        upper = np.random.randint(1, 15, size=n_apertures+1)

        year = 2020
        today = 20200131

        program_id = "GX-{}C-Q-{:03d}".format(year, program_index)

        group_index = 1
        group_id = "{:s}-{:02d}".format(program_id, group_index)

        for frame_index in range(n_frames):

            data_label = "{:s}-{:03d}".format(group_id, frame_index + 1)
            filename = "X{}S{:03d}_frame.fits".format(today, file_index)

            if frame_index == 0:
                stack_data_label = "{:s}-{:03d}_stack".format(group_id, frame_index + 1)
                stack_filename = "X{}S{:03d}_stack.fits".format(today, file_index)

            def aperture_generator(i):
                delta_center = (np.random.rand() - 0.5) * 0.9 / pixel_scale
                center[i] = center[i] + delta_center
                center[i] = np.round(center[i])
                _data = data[i]
                _error = np.random.poisson(_data) + noise_level * (np.random.rand(_data.size) - 0.5)
                _aperture = ApertureModel(
                    int(center[i]), int(lower[i]), int(upper[i]), wavelength_units, dispersion,
                    wavelength, _data, _error)
                return _aperture.__dict__

            def stack_aperture_generator(i):
                delta_center = (np.random.rand() - 0.5) * 0.9 / pixel_scale
                center[i] = center[i] + delta_center
                _data = data[i]

                _error = np.random.rand(_data.size) - 0.5
                _error *= noise_level / (frame_index + 1)
                _error += np.random.poisson(_data)

                _aperture = ApertureModel(
                    int(center[i]), int(lower[i]), int(upper[i]), wavelength_units, dispersion,
                    wavelength, _data, _error)

                return _aperture.__dict__

            n = np.random.randint(n_apertures-1, n_apertures+1)
            apertures = [aperture_generator(i) for i in range(n)]

            n = np.random.randint(n_apertures - 1, n_apertures + 1)
            stack_apertures = [stack_aperture_generator(i) for i in range(n)]

            frame = SpecPackModel(
                    data_label=data_label,
                    group_id=group_id,
                    filename=filename,
                    is_stack=False,
                    pixel_scale=pixel_scale,
                    program_id=program_id,
                    stack_size=1,
                    apertures=apertures)

            stack = SpecPackModel(
                data_label=stack_data_label,
                group_id=group_id,
                filename=stack_filename,
                is_stack=True,
                pixel_scale=pixel_scale,
                program_id=program_id,
                stack_size=frame_index + 1,
                apertures=stack_apertures)

            json_list = [frame.__dict__, stack.__dict__]

            json_data = json.dumps(json_list).encode("utf-8")

            print("\n Created JSON for single frame with: ")
            print("  Program ID: {}".format(program_id))
            print("  Group-id: {}".format(group_id))
            print("  Data-label: {}".format(data_label))
            print("  Filename: {}".format(filename))
            print("  Apertures: {}".format(center))
            print("  Performing request...")

            try:
                post_request = urllib.request.Request(url)
                post_request.add_header("Content-Type", "application/json")
                postr = urllib.request.urlopen(post_request, json_data)
                postr.read()
                postr.close()
            except urllib.error.URLError:
                import sys
                print("\n Error trying to open URL: {}".format(url))
                print(" Please, check that the server is running "
                      "and run again.\n")
                sys.exit()

            print("  Done.")
            print("  Sleeping for {} seconds ...".format(sleep_between_frames))

            time.sleep(sleep_between_frames)
            file_index += 1

        program_index += 1
        group_index += 1


def _parse_arguments():
    """
    Parses arguments received from the command line.

    Returns
    -------
    namespace
        all the default and customized options parsed from the command line.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="A script that simulates a pipeline running the "
                    "plotSpectraForQA and posting JSON data to the "
                    "ADCC server.")

    parser.add_argument(
        '-a', '--apertures',
        default=3,
        dest="n_apertures",
        help="Number of aperetures for each data",
        type=int,
    )

    parser.add_argument(
        '-f', '--frames',
        default=3,
        dest="n_frames",
        help="Number of frames for each Group ID.",
        type=int,
    )

    parser.add_argument(
        '-u', '--url',
        default=URL,
        help="URL of the ADCC server (e.g.: http://localhost:8777/spec_report)",
        type=str,
    )

    parser.add_argument(
        '-s', '--sleep',
        default=10.,
        dest="sleep_time",
        help="Sleep time between post requests",
        type=float,
    )

    return parser.parse_args()


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

    def __init__(self, data_label, group_id, filename, is_stack, pixel_scale, program_id, stack_size, apertures):

        self.data_label = data_label
        self.group_id = group_id
        self.filename = filename
        self.msgtype = "specjson"
        self.is_stack = is_stack
        self.pixel_scale = pixel_scale
        self.program_id = program_id
        self.stack_size = stack_size
        self.timestamp = time.time()

        self.apertures = apertures


if __name__ == '__main__':
    main()
