
class GMOS_IFU_BLUE(DataClassification):
    name="GMOS_IFU_BLUE"
    usage = ""
    parent = "GMOS_IFU"
    requirement = ISCLASS('GMOS_IFU') & PHU(MASKNAME='(IFU-B)|(IFU-B-NS)')

newtypes.append(GMOS_IFU_BLUE())
