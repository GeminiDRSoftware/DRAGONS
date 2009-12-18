
class GMOS_IFU_BLUE(DataClassification):
    name="GMOS_IFU_BLUE"
    usage = ""
    typeReqs= ['GMOS_IFU']
    phuReqs= {'MASKNAME': '(IFU-B)|(IFU-B-NS)'}

newtypes.append(GMOS_IFU_BLUE())
