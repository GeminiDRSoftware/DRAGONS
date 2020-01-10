#!/usr/bin/env python
"""
This script is used to develop SpecViewer only. It should not be used for
production.

It creates a JSON file from a dictionary to be used in the QAP SpecViewer
development.
"""

import json

import numpy as np


def main():
    # Create fake source
    np.random.seed(0)

    image_width = 4000
    image_height = 2000

    obj_max_weight = 300.
    obj_continnum = 300. + 0.01 * np.arange(image_width)

    # Create aperture data
    apertures = []
    for i in range(3):
        center = np.random.randint(100, image_height-100)
        lower = np.random.randint(-15, -1)
        upper = np.random.randint(1, 15)
        dispersion = 0.15
        intensity = create_1d_spectrum(
            image_width,
            int(0.1 * image_width), obj_max_weight) + obj_continnum
        variance = np.sqrt(intensity)
        wavelength = np.arange(image_width) * dispersion + 4000.

        apertures.append(
            {
                "center": center,
                "lower": lower,
                "upper": upper,
                "dispersion": dispersion,
                "wavelength": list(wavelength),
                "intensity": list(intensity),
                "variance": list(variance),
             }
        )

    # Create dict with all the data
    data = dict(filename="N20001231S001_suffix.fits", programId="GX-2000C-Q-001")
    data["apertures"] = apertures

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
