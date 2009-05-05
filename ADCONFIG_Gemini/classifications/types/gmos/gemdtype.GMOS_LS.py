
class GMOS_LONGSLIT(DataClassification):
    name="GMOS_LONGSLIT"
    usage = ""
    typeReqs= ['GMOS']
    phuReqs= {'OBSMODE': 'LONGSLIT'}

newtypes.append(GMOS_LONGSLIT())
