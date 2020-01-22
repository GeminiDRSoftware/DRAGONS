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
import json

from flask import (Blueprint, abort, jsonify, redirect, send_from_directory,
                   url_for, render_template)

from recipe_system.adcc.servers import http_proxy

qlook = Blueprint(
    'qlook',
    __name__,
    static_folder=os.path.dirname(__file__),
    static_url_path='',
    template_folder='./templates/')


@qlook.route('/')
def index():
    return redirect(url_for('qlook.specviewer'))


@qlook.route('/specviewer/')
def specviewer():
    return qlook.send_static_file("specviewer.html")


@qlook.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)


@qlook.route('/images/<path:path>')
def send_images(path):
    return send_from_directory('images', path)


@qlook.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)


@qlook.route('/specviewer/favicon.ico')
def favicon():
    return qlook.send_static_file("images/dragons_favicon.ico")


@qlook.route('/rqsite.json')
def rqsite():
    return http_proxy.server_time()


@qlook.errorhandler(500)
def server_error(e):
    return render_template("500.html", error=str(e)), 500


@qlook.route('/specqueue.json')
def specframe():

    path_to_data = os.getenv('SPEC_DATA')

    if path_to_data is None:
        abort(500, description="Environment variable not defined: SPEC_DATA")

    filename = "data.json"
    filename = os.path.join(path_to_data, filename)

    with open(filename, 'r') as json_file:
        data = json.load(json_file)

    try:
        return jsonify(json.dumps(data))
    except Exception as e:
        print(str(e))
        return jsonify(str(e))


