import inspect
import astrodata
import gemini_instruments
import re
import importlib
import sys
import os


from ..show_primitives import show_primitives

def test_show_primitives():
    answer = show_primitives()