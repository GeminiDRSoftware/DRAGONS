class GMOS_N(DataClassification):
    name="GMOS_N"
    usage = ""
    typeReqs= []
    phuReqs= {}
    parent = "GMOS"
    requirement = PHU(INSTRUME='GMOS-N')

newtypes.append(GMOS_N())
