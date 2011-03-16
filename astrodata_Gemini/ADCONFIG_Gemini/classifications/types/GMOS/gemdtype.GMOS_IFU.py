
class GMOS_IFU(DataClassification):
    name="GMOS_IFU"
    usage = "Data taken in the IFU instrument mode with either GMOS instrument"
    parent = "GMOS_SPECT"
    requirement = OR(AND(ISCLASS('GMOS_SPECT'),
                         PHU(OBSMODE='IFU',OBSTYPE='OBJECT')),
                     ISCLASS(    'GMOS_IFU_FLAT'),
                     ISCLASS('GMOS_IFU_TWILIGHT'),
                     ISCLASS(     'GMOS_IFU_ARC') )

newtypes.append(GMOS_IFU())
