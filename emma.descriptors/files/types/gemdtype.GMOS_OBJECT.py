
class GMOS_OBJECT(DataClassification):
    name="GMOS_OBJECT"
    usage = ""
    typeReqs= ['GMOS']
    phuReqs= {'OBSTYPE': 'OBJECT'}

newtypes.append(GMOS_OBJECT())
