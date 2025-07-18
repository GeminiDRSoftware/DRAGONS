"""This module contains the bokeh styles used by the interactive viewer.

A manager handles the styles and provides a way to include multiple styles.
"""
import bokeh.models as bm
from bokeh.io import curdoc


class DragonsStyleManager:
    _url = "/dragons/static/dragons.css"

    @classmethod
    def stylesheet(self):
        """Returns the stylesheet, creating a new instance if the document is
        new.

        As bokeh renders new documents, this will actually create multiple new
        ImportedStyleSheet objects, because bokeh may update/re-create the
        document. This is not an issue, since this does not change/unlink the
        previous stylesheets, and stylesheets are just information for bokeh
        to link the stylesheet in the various shadow DOMs.

        This only occurs 2-3 times per document, so it's not a big deal.
        """
        return bm.ImportedStyleSheet(url=self._url)

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
        DragonsStyleManager.stylesheet(),
    ]

    return stylesheets

def skip_btn_styles():
    """
    Returns a list of styles for the "Skip" button that overrides
    the bokeh default button_type colors.

    It appears to require an InlineStyleSheet.
    """
    skip_stylesheet = bm.InlineStyleSheet(
                css=".bk-btn {\
                        border: none; \
                        background-color: #8e7cc3; \
                        border-radius: 5px;\
                        font-size: 16px;\
                        padding: 5px 35px;\
                        transition: all .2s ease-in-out;\
                    }\n"
                    ".bk-btn:active {\
                        border: none; \
                        background-color: #8e7cc3;\
                        filter: brightness(90%);\
                        -webkit-box-shadow: 0 4px 4px rgba(0, 0, 0, 0.2) inset;\
                        -moz-box-shadow: 0 4px 4px rgba(0, 0, 0, 0.2) inset;\
                        box-shadow: 0 4px 4px rgba(0, 0, 0, 0.2) inset;\
                     }\n"
                    ".bk-btn:hover {\
                        border: none;\
                        background-color: #8e7cc3;\
                        filter: saturate(90%);\
                        box-shadow: 0px 4px 4px rgba(0, 0, 0, 0.2);\
                     }\n"
            )

    return [skip_stylesheet]