
class GMOS_MOSFLAT(DataClassification):
    name="GMOS_MOSFLAT"
    usage = ""
    typeReqs= ['GMOS_FLAT', 'GMOS_MOS']

newtypes.append(GMOS_MOSFLAT())
