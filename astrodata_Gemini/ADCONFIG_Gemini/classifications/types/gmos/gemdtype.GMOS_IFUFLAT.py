
class GMOS_IFUFLAT(DataClassification):
    name="GMOS_IFUFLAT"
    usage = ""
    parent = "GMOS_FLAT"
    requirement = ISCLASS('GMOS_FLAT', 'GMOS_IFU')

newtypes.append(GMOS_IFUFLAT())
