
class NON_SIDEREAL(DataClassification):
    name="NON_SIDEREAL"
    usage = "Data taken with the telesocope not tracking siderealy"
    
    parent = "GEMINI"
    requirement =  PHU(FRAME='FK5') & NOT(ISCLASS("SIDEREAL"))

newtypes.append(NON_SIDEREAL())
