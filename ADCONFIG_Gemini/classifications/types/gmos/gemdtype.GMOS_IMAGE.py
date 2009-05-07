
class GMOS_IMAGE(DataClassification):
    name="GMOS_IMAGE"
    usage = ""
    typeReqs= ['GMOS']
    phuReqs= {'OBSMODE': 'IMAGE',
                'OBSTYPE': 'OBJECT'
                }

newtypes.append(GMOS_IMAGE())
