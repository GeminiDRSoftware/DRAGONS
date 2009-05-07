
class GMOS_LSFLAT(DataClassification):
    name="GMOS_LSFLAT"
    usage = ""
    typeReqs= ['GMOS_FLAT', 'GMOS_LS']

newtypes.append(GMOS_LSFLAT())
