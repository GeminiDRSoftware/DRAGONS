
class GMOS_IMAGE(DataClassification):
    name="GMOS_IMAGE"
    usage = ""
    typeReqs= ['GMOS']
    phuReqs= {  
                # Imaging by definition uses the MIRROR as a grating.
                'GRATING': 'MIRROR'
                }

newtypes.append(GMOS_IMAGE())
