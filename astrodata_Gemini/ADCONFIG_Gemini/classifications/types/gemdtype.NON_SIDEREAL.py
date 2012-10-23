
class NON_SIDEREAL(DataClassification):
    name="NON_SIDEREAL"
    usage = "Data taken with the telesocope not tracking siderealy"
    
    parent = "GEMINI"
    requirement =  NOT(ISCLASS("SIDEREAL"))

newtypes.append(NON_SIDEREAL())
