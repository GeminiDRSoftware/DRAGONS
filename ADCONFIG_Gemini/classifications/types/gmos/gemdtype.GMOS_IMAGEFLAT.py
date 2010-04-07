class GMOS_IMAGEFLAT(DataClassification):
    name="GMOS_IMAGEFLAT"
    usage = ""
    parent = "GMOS_FLAT"
    requirement = ISCLASS('GMOS_FLAT', 'GMOS_IMAGE')

newtypes.append(GMOS_IMAGEFLAT())
