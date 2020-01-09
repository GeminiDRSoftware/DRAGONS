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
        "apertures": 3,
        "filename": "N20001231S001_suffix.fits",
        "programId": "GX-2000C-Q-001",
    }

    with open(filename, 'w') as json_file:
        json.dump(data, json_file)


if __name__ == '__main__':
    main()
