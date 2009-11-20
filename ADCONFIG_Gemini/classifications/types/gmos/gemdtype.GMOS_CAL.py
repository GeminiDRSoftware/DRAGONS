
class GMOS_CAL(ORClassification):
    name="GMOS_CAL"
    usage = ""
    typeOrs= [ 'GMOS',
                'GMOS_FLAT',
                'GMOS_TWILIGHT',
                'GMOS_DARK',
                'GMOS_BIAS']

newtypes.append(GMOS_CAL())
