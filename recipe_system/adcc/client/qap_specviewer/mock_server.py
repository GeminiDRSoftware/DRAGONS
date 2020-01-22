#!/usr/bin/env python
"""
This script is used to develop SpecViewer only. It should not be used for
production.

It starts a local server that returns an artificial JSON file containing data
from an imaginary FITS file that can represent a single or a stack frame. It
is using to design frontend/backend.

Remember to run it from the same folder where it lives since it relies on
relative path. This is a temporary solution.
"""

import os

from flask import Flask, send_from_directory

try:
    from .mock_qlook import qlook
except ModuleNotFoundError:
    from sys import exit
    print(' Start Flask server using the following steps: \n'
          ' $ export FLASK_APP={path to this file}\n'
          ' $ export FLASK_ENV=development\n'
          ' $ flask run')
    exit(1)


app = Flask(__name__, static_folder=os.path.dirname(__file__))
app.register_blueprint(qlook, url_prefix='/qlook')


@app.route('/')
def index():
    return app.send_static_file("index.html")


@app.route('/css/<path:path>')
def css(path):
    return send_from_directory("css", path)


if __name__ == '__main__':
    app.run()
