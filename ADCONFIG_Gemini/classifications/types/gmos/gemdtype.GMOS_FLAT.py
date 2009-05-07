
class GMOS_FLAT(DataClassification):
    name="GMOS_FLAT"
    usage = ""
    typeReqs= ['GMOS']
    phuReqs= {
                'OBSTYPE': 'FLAT'
            }

newtypes.append(GMOS_FLAT())
