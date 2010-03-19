
class GMOS_IFU_RED(DataClassification):
    name="GMOS_IFU_RED"
    usage = ""
    requirement = ISCLASS('GMOS_IFU') & PHU(MASKNAME='(IFU-R)|(IFU-R-NS)')

newtypes.append(GMOS_IFU_RED())
