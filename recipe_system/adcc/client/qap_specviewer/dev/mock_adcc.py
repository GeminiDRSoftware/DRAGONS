#!/usr/bin/env python
"""
This script is used to develop SpecViewer only. It should not be used for
production.

It starts a local server that returns an artificial JSON file containing data
from an imaginary FITS file that can represent a single or a stack frame. It
is using to design frontend/backend.
"""

import json

from flask import Flask

app = Flask(__name__)


@app.route('/getframe')
def get_frame():

    fake_dict = {
        "fruits": "apple"
    }

    return json.dumps(fake_dict)


if __name__ == '__main__':
    app.run()
