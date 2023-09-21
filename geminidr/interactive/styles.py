"""This module contains the bokeh styles used by the interactive viewer.

A manager handles the styles and provides a way to include multiple styles.
"""
import bokeh.models as bm


def dragons_styles():
    """Returns a list of fresh style sheets to be used by the interactive
    viewer.

    Notes
    -----
    This is a function, not a class, to make usage obvious and to avoid
    problems with bokeh models not being sharable across documents.
    """
    # This is just a single stylesheet. I'm keeping it in this file to be used
    # across the interactive viewer. If we need more stylesheets, we can create
    # new ImportedStyleSheets here, to keep everything in one place.
    stylesheets = [
        bm.ImportedStyleSheet(url="/dragons/static/dragons.css"),
    ]

    return stylesheets