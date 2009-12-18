
class GMOS_MOS(DataClassification):
    name="GMOS_MOS"
    usage = ""
    typeReqs= ['GMOS']
    phuReqs= {'OBSMODE': 'MOS'}

newtypes.append(GMOS_MOS())
