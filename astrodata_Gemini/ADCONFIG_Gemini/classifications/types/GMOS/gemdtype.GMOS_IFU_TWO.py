
class GMOS_IFU_TWO(DataClassification):
    name="GMOS_IFU_TWO"
    usage = ""
    parent = "GMOS_IFU"
    requirement = ISCLASS('GMOS_IFU') &  \
                 PHU(MASKNAME='(IFU-2)|(IFU-2-NS)|(g.ifu_slit._mdf)')

newtypes.append(GMOS_IFU_TWO())
