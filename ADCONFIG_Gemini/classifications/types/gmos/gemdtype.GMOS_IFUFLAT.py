
class GMOS_IFUFLAT(DataClassification):
    name="GMOS_IFUFLAT"
    usage = ""
    typeReqs= ['GMOS_FLAT', 'GMOS_IFU']

newtypes.append(GMOS_IFUFLAT())
