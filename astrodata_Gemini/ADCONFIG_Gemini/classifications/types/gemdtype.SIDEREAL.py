
class SIDEREAL(DataClassification):
    name="SIDEREAL"
    usage = "Data taken with the telesocope tracking siderealy"
    
    parent = "GEMINI"
    requirement = PHU(DECTRACK='0.') & PHU(RATRACK='0.') & PHU(FRAME='FK5')

newtypes.append(SIDEREAL())
