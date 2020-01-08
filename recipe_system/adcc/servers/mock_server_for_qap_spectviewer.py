#!/usr/bin/env python
"""
Mock Server for QAP SpecViewer

QAP SpecViewer uses POST and GET requests to retrieve data from ADCC. We do not
want to touch ADCC just yet so let me write a simple mock server that will
fulfill this role for now.
"""

from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/spec_viewer')
def spec_viewer():
    return 'Spec Viewer'


def main():
    """
    Main function used to start the Mock Server for QAP SpecViewer.
    """
    app.run()


if __name__ == '__main__':
    main()
