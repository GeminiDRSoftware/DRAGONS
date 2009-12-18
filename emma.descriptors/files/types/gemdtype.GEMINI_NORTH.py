
class GEMINI_NORTH(DataClassification):
    name="GEMINI_NORTH"
    usage = "Data taken at Gemini North upon Mauna Kea"
    typeReqs= []
    phuReqs= {'TELESCOP': 'Gemini-North', 'OBSERVAT': 'Gemini-North'}

newtypes.append(GEMINI_NORTH())
