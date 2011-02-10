
class GMOS_NODANDSHUFFLE(DataClassification):
    name="GMOS_NODANDSHUFFLE"
    usage = ""
    typeReqs= []
    phuReqs= {}
    parent = "GMOS"
    requirement = PHU(NODPIX='.*')

newtypes.append(GMOS_NODANDSHUFFLE())
