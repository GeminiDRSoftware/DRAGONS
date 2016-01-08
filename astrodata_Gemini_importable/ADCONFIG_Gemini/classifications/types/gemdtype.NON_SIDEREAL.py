
class NON_SIDEREAL(DataClassification):
    name="NON_SIDEREAL"
    usage = "Data taken with the telesocope not tracking siderealy"
    
    parent = "GEMINI"
    requirement = AND([
                      OR([PHU(TRKFRAME="FK5"), PHU(FRAME="FK5"), PHU(TRKFRAME="APPT"), PHU(FRAME="APPT")]),
                      NOT(ISCLASS("SIDEREAL"))
                     ])

newtypes.append(NON_SIDEREAL())
