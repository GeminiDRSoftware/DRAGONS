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

from flask import Flask, send_from_directory, jsonify

from recipe_system.adcc.servers import http_proxy

app = Flask(__name__,
            static_folder=os.path.dirname(__file__),
            static_url_path='',
            root_path=os.path.dirname(__file__))


@app.route('/')
def root():
    return app.send_static_file("index.html")


@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)


@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)


@app.route('/<path:path>')
def get_file(path):
    return app.send_static_file(path)


@app.route('/specviewer/')
def specviewer():
    return app.send_static_file("specviewer.html")


@app.route('/rqsite.json')
def rqsite():
    return http_proxy.server_time()


@app.route('/specframe.json')
def specframe():
    filename = "data.json"

    with open(filename, 'r') as json_file:
        data = json.load(json_file)

    try:
        return jsonify(json.dumps(data))
    except Exception as e:
        print(str(e))
        return jsonify(str(e))


if __name__ == '__main__':
    app.run()
