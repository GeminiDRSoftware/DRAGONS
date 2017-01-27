import os

import astrodata
import gemini_instruments

THIS_DIR = os.path.dirname(__file__)
CHARA = '/net/chara/data2/pub'

def from_test_data(fname):
    return astrodata.open(os.path.join(THIS_DIR, 'test_data', fname))

def from_chara(fname):
    return astrodata.open(os.path.join(CHARA, fname))

