
class GMOS_DARK(DataClassification):
    name="GMOS_DARK"
    usage = ""
    typeReqs= ['GMOS']
    phuReqs= {
                'OBSTYPE': 'DARK'
            }

newtypes.append(GMOS_DARK())
