#!/usr/bin/env python

from flask import Blueprint, redirect, render_template, send_from_directory, url_for

react_sviewer = Blueprint(
    'react_sviewer', __name__,
    template_folder="templates", static_folder="static", static_url_path="/")


@react_sviewer.route('/')
def index():
    return redirect(url_for('react_sviewer.specviewer'))


@react_sviewer.route('/specviewer/')
def specviewer():
    return render_template("rspecviewer.html")


@react_sviewer.route('/css/<path:path>')
def send_css(path):
    # Todo - Find proper way to deal with static files
    return react_sviewer.send_static_file("css/{}".format(path))


@react_sviewer.route('/images/<path:path>')
def send_images(path):
    # Todo - Find proper way to deal with static files
    return react_sviewer.send_static_file("images/{}".format(path))


@react_sviewer.route('/js/<path:path>')
def send_js(path):
    # Todo - Find proper way to deal with static files
    return react_sviewer.send_static_file("js/{}".format(path))

