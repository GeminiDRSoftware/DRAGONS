
class GMOS_CAL(DataClassification):
    name="GMOS_CAL"
    usage = ""
    parent = "GMOS"
    requirement = ISCLASS('GMOS') & OR( ISCLASS('   GMOS_IM_FLAT'),
                                        ISCLASS(' GMOS_IM_TWFLAT'),
                                        ISCLASS('      GMOS_DARK'),
                                        ISCLASS('   GMOS_LS_FLAT'),
                                        ISCLASS(' GMOS_LS_TWFLAT'),
                                        ISCLASS('  GMOS_MOS_FLAT'),
                                        ISCLASS('GMOS_MOS_TWFLAT'),
                                        ISCLASS('  GMOS_IFU_FLAT'),
                                        ISCLASS('GMOS_IFU_TWFLAT'),
                                        ISCLASS('      GMOS_BIAS'), )

newtypes.append(GMOS_CAL())
