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
from flask import Blueprint, redirect, send_from_directory, url_for

from recipe_system.adcc.servers import http_proxy

qlook = Blueprint(
    'qlook',
    __name__,
    static_folder='',
    template_folder='./templates/')


@qlook.route('/')
def index():
    return redirect(url_for('qlook.specviewer'))


@qlook.route('/specviewer/')
def specviewer():
    return qlook.send_static_file("specviewer.html")


@qlook.route('/css/<path:path>')
def send_css(path):
    # return send_from_directory('css', path)  # Todo - Understand why it fails
    return qlook.send_static_file("css/{}".format(path))


@qlook.route('/images/<path:path>')
def send_images(path):
    # return send_from_directory('images', path)  # Todo - Understand why it fails
    return qlook.send_static_file("images/{}".format(path))


@qlook.route('/js/<path:path>')
def send_js(path):
    # return send_from_directory('/specviewer/js', path)  # Todo - Understand why it fails
    return qlook.send_static_file("js/{}".format(path))


@qlook.route('/specviewer/favicon.ico')
def favicon():
    return qlook.send_static_file("images/dragons_favicon.ico")


@qlook.route('/rqsite.json')
def rqsite():
    return http_proxy.server_time()
