import astrodata
import instruments

def test_gemini_with_no_instrument():
    # This should
    obj = astrodata.create(phu={'OBSERVAT': 'GEMINI-NORTH', 'TELESCOP': 'GEMINI'})
    assert isinstance(obj, instruments.gemini.AstroDataGemini)
