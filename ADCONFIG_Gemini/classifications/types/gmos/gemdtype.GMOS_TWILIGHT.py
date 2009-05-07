
class GMOS_TWILIGHT(DataClassification):
    name="GMOS_TWILIGHT"
    usage = ""
    typeReqs= ['GMOS']
    phuReqs= {
                'OBJECT': 'Twilight',
            }

newtypes.append(GMOS_TWILIGHT())
