#!/usr/bin/env python
"""
This script is used to develop SpecViewer only. It should not be used for
production.

It creates a JSON file from a dictionary to be used in the QAP SpecViewer
development.
"""

import json


def main():
    filename = "data.json"

    data = {
        "filename": "N20001231S001_suffix.fits",
        "programId": "GX-2000C-Q-001",
    }

    apertures = [
        {
            "center": 515,
            "lower": -5,
            "upper": +10,
            "dispersion": 0.15,
        }, {
            "center": 123,
            "lower": -7,
            "upper": +15,
            "dispersion": 0.15,
        }, {
            "center": 321,
            "lower": -14,
            "upper": +12,
            "dispersion": 0.15,
        },
    ]

    data["apertures"] = apertures

    with open(filename, 'w') as json_file:
        json.dump(data, json_file)


if __name__ == '__main__':
    main()
