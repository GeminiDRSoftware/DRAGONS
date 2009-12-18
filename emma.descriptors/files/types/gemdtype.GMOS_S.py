
class GMOS_S(DataClassification):
    name="GMOS_S"
    usage = "For data from GMOS South"
    typeReqs= []
    phuReqs= {'INSTRUME': 'GMOS-S'}

newtypes.append(GMOS_S())
