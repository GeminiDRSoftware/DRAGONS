
class GMOS_BIAS(DataClassification):
    name="GMOS_BIAS"
    usage = ""
    typeReqs= ['GMOS']
    phuReqs= {
                'OBSTYPE': 'BIAS'
            }

newtypes.append(GMOS_BIAS())
