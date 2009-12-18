
class GEMINI_SOUTH(DataClassification):
    name="GEMINI_SOUTH"
    usage = ""
    typeReqs= []
    phuReqs= {'TELESCOP': 'Gemini-South', 'OBSERVAT': 'Gemini-South'}

newtypes.append(GEMINI_SOUTH())
