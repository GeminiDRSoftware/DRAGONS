#!/usr/bin/env python
"""
This script is used to develop SpecViewer only. It should not be used for
production.

It creates a JSON file from a dictionary to be used in the QAP SpecViewer
development.
"""

import json

import numpy as np
from scipy.ndimage import gaussian_filter1d


def main():
    # Create fake source
    np.random.seed(0)

    width = 4000
    height = 2000

    min_wavelength = 4000.
    max_wavelength = 7000.

    obj_max_weight = 3000.
    obj_continnum = 300. + 0.01 * np.arange(width)
    noise = 30
    dispersion = 0.12  # nm / px

    # Create aperture data
    apertures = []
    stack_apertures = []

    for i in range(3):
        center = np.random.randint(100, height - 100)
        lower = np.random.randint(-15, -1)
        upper = np.random.randint(1, 15)

        intensity = create_1d_spectrum(
            width,
            int(0.01 * width), obj_max_weight) + obj_continnum
        intensity = gaussian_filter1d(intensity, 5)

        variance = 0.1 * (np.random.poisson(intensity) +
           noise * (np.random.rand(width) - 0.5))

        wavelength = min_wavelength + np.arange(width) * dispersion * 10

        aperture = {
            "apertureId": i,
            "center": center,
            "lower": lower,
            "upper": upper,
            "dispersion": dispersion,
            "wavelength": list(wavelength),
            "intensity": list(intensity),
            "variance": list(variance),
        }

        stack_aperture = {
            "wavelength": list(wavelength),
            "intensity": list(intensity),
            "variance": list(variance / 10),
        }

        apertures.append(aperture)
        stack_apertures.append(stack_aperture)

    # Create dict with all the data
    data = {"msgtype": "specviewer",
            "filename": "N20001231S001_suffix.fits",
            "programId": "GX-2000C-Q-001",
            "apertures": apertures,
            "stackApertures": stack_apertures}

    filename = "data.json"
    with open(filename, 'w') as json_file:
        json.dump(data, json_file)


def create_1d_spectrum(width, n_lines, max_weight):
    """
    Generates a 1D NDArray with the sky spectrum.

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

    return spectrum


if __name__ == '__main__':
    main()
