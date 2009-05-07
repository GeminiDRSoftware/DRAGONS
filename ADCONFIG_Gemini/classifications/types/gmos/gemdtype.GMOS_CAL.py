
class GMOS_CAL(DataClassification):
    name="GMOS_CAL"
    usage = ""
    typeReqs= [ 'GMOS',
                'GMOS_FLAT',
                'GMOS_TWILIGHT',
                'GMOS_DARK',
                'GMOS_BIAS']

newtypes.append(GMOS_CAL())
