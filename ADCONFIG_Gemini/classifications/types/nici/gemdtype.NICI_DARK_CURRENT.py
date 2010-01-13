
class NICI_DARK_CURRENT(DataClassification):
    name="NICI_DARK_CURRENT"
    usage = ""
    typeReqs= ['NICI']
    phuReqs= {'OBSTYPE': 'DARK' }

newtypes.append(NICI_DARK_CURRENT())
