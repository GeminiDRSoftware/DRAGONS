
class GMOS_IMAGE(DataClassification):
    name="GMOS_IMAGE"
    usage = ""
    typeReqs= ['GMOS']
    phuReqs= {  
                # check --> current GMOS data does not have this 'OBSMODE': 'IMAGE',
                'OBSTYPE': 'OBJECT'
                }

newtypes.append(GMOS_IMAGE())
