
class GMOS_CAL(DataClassification):
    name="GMOS_CAL"
    usage = ""
    parent = "GMOS"
    requirement = ISCLASS('GMOS') & OR( ISCLASS(          'GMOS_IMAGE_FLAT'),
                                        ISCLASS(      'GMOS_IMAGE_TWILIGHT'),
                                        ISCLASS(                'GMOS_DARK'),
                                        ISCLASS(             'GMOS_LS_FLAT'),
                                        ISCLASS(         'GMOS_LS_TWILIGHT'),
                                        ISCLASS(            'GMOS_MOS_FLAT'),
                                        ISCLASS(        'GMOS_MOS_TWILIGHT'),
                                        ISCLASS(            'GMOS_IFU_FLAT'),
                                        ISCLASS(        'GMOS_IFU_TWILIGHT'),
                                        ISCLASS(                'GMOS_BIAS') )

newtypes.append(GMOS_CAL())
