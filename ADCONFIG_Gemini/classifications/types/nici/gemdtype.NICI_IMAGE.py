
class NICI_IMAGE(DataClassification):
    name="NICI_IMAGE"
    usage = ""
    typeReqs= ['NICI']
    phuReqs= {'INSTRUME': 'NICI'}

newtypes.append(NICI_IMAGE())
