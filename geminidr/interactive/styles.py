"""This module contains the bokeh styles used by the interactive viewer.

A manager handles the styles and provides a way to include multiple styles.
"""
import bokeh.models as bm


# This is just a single stylesheet. I'm keeping it in this file to be used
# across
dragons_styles = [
    bm.ImportedStyleSheet(url="/dragons/static/dragons.css"),
]