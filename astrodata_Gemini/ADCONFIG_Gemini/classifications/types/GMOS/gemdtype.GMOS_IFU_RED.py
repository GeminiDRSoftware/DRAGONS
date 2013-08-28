
class GMOS_IFU_RED(DataClassification):
    name="GMOS_IFU_RED"
    usage = ""
    parent = "GMOS_IFU"
    requirement = ISCLASS('GMOS_IFU') & \
                  PHU(MASKNAME='(IFU-R)|(IFU-R-NS)|(g.ifu_slit._mdf)' ) 
                   

newtypes.append(GMOS_IFU_RED())
