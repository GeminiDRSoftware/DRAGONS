
class GMOS_IMAGE(DataClassification):
    name="GMOS_IMAGE"
    usage = ""
    typeReqs= ['GMOS']
    phuReqs= {'OBSMODE': 'IMAGE'}

newtypes.append(GMOS_IMAGE())
