
class GMOS_IFU_RED(DataClassification):
    name="GMOS_IFU_RED"
    usage = ""
    typeReqs= ['GMOS_IFU']
    phuReqs= {'MASKNAME': '(IFU-R)|(IFU-R-NS)'}

newtypes.append(GMOS_IFU_RED())
