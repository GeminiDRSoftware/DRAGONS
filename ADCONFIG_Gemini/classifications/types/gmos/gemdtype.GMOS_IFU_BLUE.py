
class GMOS_IFU_BLUE(DataClassification):
    name="GMOS_IFU_BLUE"
    usage = ""
    requirement = ISCLASS('GMOS_IFU') & PHU(MASKNAME='(IFU-B)|(IFU-B-NS)')

newtypes.append(GMOS_IFU_BLUE())
