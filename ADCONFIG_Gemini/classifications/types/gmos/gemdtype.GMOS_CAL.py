
class GMOS_CAL(DataClassification):
    name="GMOS_CAL"
    usage = ""
    requirement = ISCLASS('GMOS') & OR( ISCLASS('GMOS_FLAT'),
                                        ISCLASS('GMOS_TWILIGHT'),
                                        ISCLASS('GMOS_DARK'),
                                        ISCLASS('GMOS_BIAS'))

newtypes.append(GMOS_CAL())
