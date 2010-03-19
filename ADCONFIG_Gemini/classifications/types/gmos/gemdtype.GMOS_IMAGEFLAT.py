class GMOS_IMAGEFLAT(DataClassification):
    name="GMOS_IMAGEFLAT"
    usage = ""
    requirement = ISCLASS('GMOS_FLAT', 'GMOS_IMAGE')

newtypes.append(GMOS_IMAGEFLAT())
