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
import json
import os

from flask import Flask, abort, jsonify, render_template, send_from_directory

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


@app.route('/specqueue.json')
def specframe():

    path_to_data = os.getenv('SPEC_DATA')

    if path_to_data is None:
        abort(500, description="Environment variable not defined: SPEC_DATA")

    with open(path_to_data, 'r') as json_file:
        jdata = json.load(json_file)

    try:
        # return jsonify(json.dumps(jdata))
        return jdata
    except Exception as e:
        print(str(e))
        return jsonify(str(e))


@qlook.errorhandler(500)
def server_error(e):
    return render_template("500.html", error=str(e)), 500



if __name__ == '__main__':
    app.run()
