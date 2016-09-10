
class SIDEREAL(DataClassification):
    name="SIDEREAL"
    usage = "Data taken with the telesocope tracking siderealy"
    
    parent = "GEMINI"
#    requirement = PHU(DECTRACK='0.') & PHU(RATRACK='0.') & PHU(FRAME='FK5')
    requirement = AND([
                      OR([PHU(TRKFRAME="FK5"), PHU(FRAME="FK5"), PHU(TRKFRAME="APPT"), PHU(FRAME="APPT")]),
                      OR([PHU(DECTRACK="^0$"), PHU(DECTRACK="^0.(0)*\s*$"),]),
                      OR([PHU(RATRACK="^0$"), PHU(RATRACK="^0.(0)*\s*$")])
                     ])   

newtypes.append(SIDEREAL())
