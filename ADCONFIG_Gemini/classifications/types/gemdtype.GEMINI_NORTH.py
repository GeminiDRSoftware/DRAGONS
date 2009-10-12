
class GEMINI_NORTH(DataClassification):
    name="GEMINI_NORTH"
    usage = "Data taken at Gemini North upon Mauna Kea"
    phuReqs= {'TELESCOP': 'Gemini-North', 'OBSERVAT': 'Gemini-North'}

newtypes.append(GEMINI_NORTH())
