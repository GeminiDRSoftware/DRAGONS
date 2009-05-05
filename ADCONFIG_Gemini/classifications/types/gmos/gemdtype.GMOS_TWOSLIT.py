
class GMOS_TWOSLIT(DataClassification):
    name="GMOS_TWOSLIT"
    usage = ""
    typeReqs= ['GMOS_IFU']
    phuReqs= {'MASKNAME': '(IFU-2)|(IFU-2-NS)'}

newtypes.append(GMOS_TWOSLIT())
