# QAP SpecViewer

This folder the QAP SpecViewer, a client written in HTML/CSS/JS that makes
HTTP/GET requests to the ADCC Server periodically in order to retried a JSON
file containing the extracted spectra that will be displayed on a browser.

## Requirements

Here are the required packages in order to proper display and run the QAP
SpecViewer client:

- JQuery v1.0.9
- JQuery-UI v1.12.1
- JQPlot v3.4.1

These JavaScript packages are stored inside `./js/` folder.

It is important to notice that the downloaded JQPlot raises an error on
double-clicking events to reset the zoom of the plots. This can be fixed by
replacing all the `sel().collapse();` calls by `sel().collapse(null);` in the
`jqplot.cursor.js` file.

## Running a mock server

Right now, the ADCC Server is not configured to handle requests from QAP
SpecViewer. Because of that, a mock server was created to handle them.

This mock server uses [flask v1.1.1](http://flask.palletsprojects.com/en/1.1.x/)
which can be installed via `conda` or via `pip` commands. Once you have it, you
can run the server using the command lines below:

```shell script
  $ export FLASK_APP=${DRAGONS_REPO}/recipe_system/adcc/client/qap_specviewer/mock_adcc.py
  $ cd ${DRAGONS_REPO}/recipe_system/adcc/client/qap_specviewer/
  $ flask run
```

Where `${DRAGONS_REPO}` is the directory to where DRAGONS is cloned.

You will also need to generate a mock JSON file that will be delivered the QAP
SpecViewer. For that, simply run the command line below:

```shell script
    $ python ${DRAGONS_REPO}/recipe_system/adcc/client/qap_specviewer/mock_json.py
```
