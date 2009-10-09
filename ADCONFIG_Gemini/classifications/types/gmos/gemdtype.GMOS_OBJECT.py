
class GMOS_OBJECT(DataClassification):
    name="GMOS_OBJECT"
    usage = ""
    typeReqs= ['GMOS']
    
    phuReqs= {  'NOHEAD':"DO NOT DETECT, TEMPORARY",
                'OBSTYPE': 'OBJECT'}

newtypes.append(GMOS_OBJECT())
