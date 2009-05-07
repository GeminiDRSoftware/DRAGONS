
class GMOS_IMAGEFLAT(DataClassification):
    name="GMOS_IMAGEFLAT"
    usage = ""
    typeReqs= ['GMOS_FLAT', 'GMOS_IMAGE']

newtypes.append(GMOS_IMAGEFLAT())
