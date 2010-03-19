
class GEMINI_SOUTH(DataClassification):
    name="GEMINI_SOUTH"
    usage = ""
    
    parent = "GEMINI"
    requirement = PHU(TELESCOP='Gemini-South',OBSERVAT='Gemini-South')

newtypes.append(GEMINI_SOUTH())
