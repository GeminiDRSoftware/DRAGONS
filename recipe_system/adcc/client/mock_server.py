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
import datetime
import json
import os
import time

from flask import Flask, abort, jsonify, request, render_template, send_from_directory


try:
    from .qap_specviewer.mock_qlook import qlook
except ModuleNotFoundError:
    from sys import exit
    print(' Start Flask server using the following steps: \n'
          ' $ export FLASK_APP={path to this file}\n'
          ' $ export FLASK_ENV=development\n'
          ' $ flask run')
    exit(1)


app = Flask(__name__, static_folder='./', template_folder="./qap_specviewer/templates/")
app.register_blueprint(qlook, url_prefix='/qlook')

JSON_DATA = ""


@app.route('/')
def index():
    return app.send_static_file("index.html")


@app.route('/favicon.ico')
def favicon():
    return send_from_directory("images", "dragons_favicon.ico")


@app.route('/rqsite.json')
def get_site_information():

    lt_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    unxtime = time.time()
    utc_now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S.%f")
    utc_offset = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) - datetime.datetime.now()

    site_information = {
        "local_site": "gemini-south",
        "lt_now": lt_now,
        "tzname": time.strftime("%Z"),
        "unxtime": unxtime,
        "utc_now": utc_now,
        "utc_offset": int(utc_offset.seconds // 3600.),
    }

    return json.dumps(site_information).encode("utf-8")


@app.route('/specqueue.json')
def get_json_data():

    global JSON_DATA

    message_type = request.args.get('msgtype')

    if message_type.strip().lower() == 'specjson':
        return jsonify(JSON_DATA)
    else:
        return "Invalid message type"


@app.route('/spec_report', methods=['POST'])
def post_json_data():
    global JSON_DATA

    JSON_DATA = request.json

    return ""


@qlook.errorhandler(500)
def server_error(e):
    return render_template("500.html", error=str(e)), 500


if __name__ == '__main__':
    app.run()
